"""Deployment of the trainable-only checkpoints produced by vla-v4/scripts/train.py.

Keep model construction, image processing and normalization in the saved training
sources. A checkpoint is never reinterpreted as a different upstream HF format.
"""
from collections import deque
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import time
import os
from pathlib import Path
import shutil
import sys
from threading import Lock
import uuid

import numpy as np

from .contract import CAMERAS, JOINTS, digest, read_dataset_contract, write_json

FORMAT = 'vla_v4_trainable_v1'
HORIZONS = dict(xvla=30, flower=20, hamlet=50, rdt=64, pi05=50)
MODES = dict(xvla='mean_std', flower='bounds', hamlet='bounds', rdt='mean_std', pi05='quantiles')
ORDER = ('overhead', 'wrist', 'third')
HISTORY = dict(xvla=(0,), flower=(0,), hamlet=(-48, -32, -16, 0), rdt=(-1, 0), pi05=(0,))
BASES = dict(xvla=('xvla',), flower=('flower', 'florence'), hamlet=('hamlet',),
             rdt=('rdt', 'siglip'), pi05=('pi05', 'paligemma_tokenizer'))


def recipe(config, audit):
    model = config['model']
    if (model not in HORIZONS or config['action_horizon'] != HORIZONS[model]
            or config['normalization_mode'] != MODES[model]
            or config['cameras'] != [f'observation.images.{c}' for c in ORDER]
            or config['data_sha256'] != audit['data_sha256']
            or audit['fps'] != 30 or audit['joint_names'] != JOINTS):
        raise ValueError('Training checkpoint does not match the five-model v4 recipe')
    stats = audit['normalization']
    for key in ('observation.state', 'action'):
        for field in ('mean', 'std', 'min', 'max', 'q01', 'q99'):
            value = np.asarray(stats[key][field])
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid checkpoint statistics: {key}.{field}')
        for lo, hi in (('min', 'max'), ('q01', 'q99')):
            if np.any(np.asarray(stats[key][lo]) > stats[key][hi]):
                raise ValueError(f'Reversed checkpoint statistics: {key}.{lo}/{hi}')
        if np.any(np.asarray(stats[key]['std']) < 0):
            raise ValueError('Negative checkpoint standard deviation')
    kwargs = {}
    if model == 'pi05':
        kwargs = dict(lora=bool(config.get('lora', False)), lora_r=int(config.get('lora_r', 32)),
                      lora_alpha=int(config.get('lora_alpha', 64)), rtc_max_delay=int(config.get('rtc_max_delay', 0)))
        if not 0 <= kwargs['rtc_max_delay'] < HORIZONS[model]:
            raise ValueError('Invalid saved RTC training delay')
    image_size = config.get('image_size')
    if image_size is None:
        image_size = 384 if model == 'rdt' else 224
    dimensions = [image_size, image_size] if type(image_size) is int else image_size
    if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2
            or any(type(n) is not int or n < 1 for n in dimensions)
            or (type(image_size) is not int and model != 'hamlet')):
        raise ValueError('Invalid saved image dimensions')
    return dict(model=model, horizon=HORIZONS[model], fps=30, joint_names=JOINTS,
                action_units='rad', camera_order=list(ORDER), image_delta_indices=list(HISTORY[model]),
                image_size=image_size,
                resize='bilinear_antialias_center_black_pad', normalization=MODES[model],
                normalization_min_scale=.001, adapter_kwargs=kwargs)


def snapshot_training_runtime(checkpoint, config, audit, training_root):
    """Called by the trainer before COMPLETE, without loading a second model."""
    checkpoint, root = Path(checkpoint), Path(training_root)
    saved = recipe(config, audit)
    runtime = checkpoint / 'runtime'
    runtime.mkdir(exist_ok=False)
    for name in ('adapters.py', 'v4_data.py'):
        source = root / 'scripts' / name
        if config['source_sha256'].get(f'scripts/{name}') != digest(source):
            raise ValueError(f'Training source changed during this run: {name}')
        shutil.copy2(source, runtime / name)
    write_json(checkpoint / 'training_contract.json', saved)


def runtime_assets(root, model):
    """Hash the actual frozen weights and runtime sources, excluding datasets/optimizers."""
    root = Path(root)
    paths = []
    for name in BASES[model]:
        base = root / 'weights' / name
        if not base.is_dir():
            raise FileNotFoundError(base)
        paths += [p for p in base.rglob('*') if p.is_file() and '.cache' not in p.parts and '.parts' not in p.parts]
    upstream = root / 'models' / ('lerobot/src' if model in ('xvla', 'pi05') else f'{model}/upstream')
    if not upstream.is_dir():
        raise FileNotFoundError(upstream)
    paths += [p for p in upstream.rglob('*') if p.is_file() and p.suffix != '.pyc'
              and not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(upstream).parts)]
    if model == 'rdt':
        paths.append(root / 'reports/rdt_language.pt')
    if model == 'hamlet':
        paths.append(root / 'fr3_vla/hamlet_inference.py')
    helper = root / 'deployment_runtime/franka_lerobot_recorder'
    if helper.is_dir():
        paths += list(helper.glob('*.py'))
    return {str(p.relative_to(root)): digest(p) for p in sorted(set(paths))}


def export_checkpoint(checkpoint, output, training_root, dataset_root, replan_steps=4):
    checkpoint, output, root = Path(checkpoint), Path(output), Path(training_root).resolve(strict=True)
    if not (checkpoint / 'COMPLETE').is_file():
        raise ValueError('Refusing an incomplete training checkpoint')
    config = json.loads((checkpoint / 'config.json').read_text())
    audit = json.loads((checkpoint / 'data_audit.json').read_text())
    inference = recipe(config, audit)
    contract, _ = read_dataset_contract(dataset_root)
    data_hash = hashlib.sha256()
    for path in sorted((Path(dataset_root) / 'data').rglob('*.parquet')):
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                data_hash.update(block)
    source_hashes = {audit['data_sha256']}
    source_hashes.update(source['data_sha256'] for source in audit.get('datasets', []))
    if data_hash.hexdigest() not in source_hashes:
        raise ValueError('Export dataset contents differ from the training checkpoint')
    if type(replan_steps) is not int or not 1 <= replan_steps <= min(14, inference['horizon']):
        raise ValueError('Replan interval must fit the 500 ms executor budget')
    import pandas as pd
    tasks = pd.read_parquet(Path(dataset_root) / 'meta/tasks.parquet')
    if len(tasks) != 1:
        raise ValueError('Expected one original v4 task')
    if output.exists():
        raise FileExistsError(output)
    assets = runtime_assets(root, config['model'])
    output.mkdir(parents=True)
    # Copy to a new export: the original checkpoint and its optimizer remain untouched.
    for name in ('config.json', 'data_audit.json', 'trainable.safetensors', 'COMPLETE'):
        shutil.copy2(checkpoint / name, output / name)
    runtime = output / 'runtime'
    runtime.mkdir()
    has_snapshot = (checkpoint / 'training_contract.json').is_file()
    if has_snapshot and json.loads((checkpoint / 'training_contract.json').read_text()) != inference:
        raise ValueError('Saved training contract disagrees with checkpoint configuration')
    for name in ('adapters.py', 'v4_data.py'):
        source = checkpoint / 'runtime' / name if has_snapshot else root / 'scripts' / name
        if has_snapshot and digest(source) != config['source_sha256'][f'scripts/{name}']:
            raise ValueError(f'Saved training source was modified: {name}')
        shutil.copy2(source, runtime / name)
    write_json(output / 'training_contract.json', inference)
    packages = {}
    for name in ('torch', 'torchvision', 'transformers', 'peft', 'diffusers', 'timm'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = dict(schema_version=1, checkpoint_format=FORMAT, model=config['model'], fps=30,
                    joint_names=JOINTS, action_units='rad', action_definition=contract['action_definition'],
                    camera_order=list(CAMERAS), cameras=contract['cameras'],
                    contract_sha256=digest(Path(dataset_root) / 'recording/contract.json'),
                    dataset_sha256=audit['data_sha256'], task=str(tasks.index[0]),
                    horizon=inference['horizon'], replan_steps=replan_steps,
                    inference=inference, statistics=audit['normalization'],
                    assets={'training_root': str(root)}, assets_sha256={'training_root': assets},
                    runtime_packages=packages, training_source_sha256=config.get('source_sha256', {}),
                    legacy_runtime_migration=not has_snapshot,
                    status='exported_requires_offline_parity_check')
    manifest['weights_sha256'] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'fr3_deployment.json', manifest)
    return manifest


def validate_export(checkpoint, manifest):
    config = json.loads((checkpoint / 'config.json').read_text())
    audit = json.loads((checkpoint / 'data_audit.json').read_text())
    expected = recipe(config, audit)
    if (manifest['inference'] != expected or manifest['horizon'] != expected['horizon']
            or manifest['model'] != expected['model'] or manifest['statistics'] != audit['normalization']
            or manifest['dataset_sha256'] != audit['data_sha256']
            or json.loads((checkpoint / 'training_contract.json').read_text()) != expected):
        raise ValueError('Deployment recipe/statistics differ from the training checkpoint')
    required = {'COMPLETE', 'config.json', 'data_audit.json', 'trainable.safetensors',
                'training_contract.json', 'runtime/adapters.py', 'runtime/v4_data.py'}
    if not required <= manifest['weights_sha256'].keys() or not manifest['assets_sha256'].get('training_root'):
        raise ValueError('Cloud export is missing sealed weights, sources or base assets')


def require_motion_parity(checkpoint):
    checkpoint = Path(checkpoint)
    path = checkpoint / 'offline_parity.json'
    if not path.is_file():
        raise ValueError('Run scripts/check_cloud_parity.py before enabling motion for a cloud export')
    report = json.loads(path.read_text())
    if (report.get('status') != 'passed' or report.get('parity_status') != 'passed'
            or report.get('motion_admission') != 'passed'
            or report.get('manifest_sha256') != digest(checkpoint / 'fr3_deployment.json')):
        raise ValueError('Cloud checkpoint has no matching successful parity/action-admission report')


class HistoryNotReady(RuntimeError):
    """A current frame has no sufficiently close observation at a training offset."""


class FrameHistory:
    """Sample real observation history at training offsets, including episode-start padding."""
    def __init__(self, offsets):
        self.offsets = tuple(offsets)
        self.frames = deque(maxlen=max(90, 2 - min(offsets)))
        self.lock = Lock()
        self.start = None

    def clear(self):
        with self.lock:
            self.frames.clear()
            self.start = None

    def observe(self, images, stamp):
        if len(images) != 3 or not math.isfinite(stamp):
            raise ValueError('History requires three RGB frames and a finite timestamp')
        with self.lock:
            if self.frames and stamp < self.frames[-1][0]:
                raise ValueError('Observation timestamps regressed')
            if (self.frames and stamp == self.frames[-1][0]
                    and all(a is b for a, b in zip(self.frames[-1][1], images, strict=True))):
                return
            if self.start is None:
                self.start = stamp
            self.frames.append((stamp, tuple(images)))

    def select(self, current):
        if self.offsets == (0,):
            return [tuple(current)]
        with self.lock:
            snapshots, start = list(self.frames), self.start
        anchor = next((i for i in range(len(snapshots)-1, -1, -1)
                       if all(a is b for a, b in zip(snapshots[i][1], current, strict=True))), None)
        if anchor is None:
            raise RuntimeError('Current observation is absent from camera history')
        snapshots = snapshots[:anchor+1]
        result = []
        for offset in self.offsets:
            if offset == 0:
                result.append(tuple(current))
                continue
            target = snapshots[-1][0] + offset / 30
            if target < start:
                chosen = snapshots[0]
                if chosen[0] != start:
                    raise RuntimeError('Episode-start camera frame was evicted')
            else:
                chosen = min(snapshots, key=lambda item: abs(item[0] - target))
                if abs(chosen[0] - target) > .5 / 30 + 1e-6:
                    raise HistoryNotReady('Camera history has a gap at a required training offset')
            result.append(chosen[1])
        return result


def load_sources(checkpoint, root):
    """Import a sealed training snapshot, relocating only asset paths."""
    tag = '_fr3_training_' + uuid.uuid4().hex
    def load(name):
        spec = importlib.util.spec_from_file_location(tag + name, checkpoint / 'runtime' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    data = load('v4_data')
    data.ROOT = root
    previous = sys.modules.get('v4_data')
    sys.modules['v4_data'] = data
    try:
        adapters = load('adapters')
    finally:
        if previous is None:
            del sys.modules['v4_data']
        else:
            sys.modules['v4_data'] = previous
    return data, adapters


class CloudBackend:
    def __init__(self, agent):
        import torch
        from safetensors.torch import load_file
        self.device, self.task = agent.device, agent.manifest['task']
        self.recipe = agent.manifest['inference']
        root = agent.asset('training_root')
        for name, expected in agent.manifest['runtime_packages'].items():
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise ValueError(f'Cloud runtime requires {name}=={expected}, found {actual}; use the matching training environment')
        sys.path.insert(0, str(root / 'models/lerobot/src'))
        self.data, adapters = load_sources(agent.checkpoint, root)
        model = agent.model_name
        if (self.data.HORIZONS[model] != self.recipe['horizon'] or self.data.MODES[model] != self.recipe['normalization']
                or self.data.CAMERAS != [f'observation.images.{c}' for c in ORDER]):
            raise ValueError('Saved adapter sources disagree with the deployment contract')
        self.norm = self.data.Normalizer(agent.manifest['statistics'], self.recipe['normalization'])
        if model == 'hamlet':
            from transformers.modeling_utils import no_init_weights
            # Base and trained checkpoints replace every parameter before inference.
            with no_init_weights():
                self.model = adapters.ADAPTERS[model](**self.recipe['adapter_kwargs'])
        else:
            self.model = adapters.ADAPTERS[model](**self.recipe['adapter_kwargs'])
        params = dict(self.model.named_parameters())
        state = load_file(str(agent.checkpoint / 'trainable.safetensors'))
        expected = {name for name, value in params.items() if value.requires_grad}
        if set(state) != expected:
            raise ValueError(f'Trainable parameter mismatch: missing={sorted(expected-set(state))[:10]}, extra={sorted(set(state)-expected)[:10]}')
        if model == 'hamlet':
            uncovered = {f'model.{name}' for name in self.model.load_report['missing']} - set(state)
            if uncovered:
                raise ValueError(f'Uninitialized HAMLET checkpoint entries: {sorted(uncovered)[:10]}')
        with torch.no_grad():
            for name, value in state.items():
                if params[name].shape != value.shape or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid trained tensor: {name}')
                params[name].copy_(value)
        del state
        self.model.to(self.device).eval().requires_grad_(False)
        self.state_center, self.state_scale = self.norm.parameters('observation.state', torch.empty(7))
        self.action_center, self.action_scale = self.norm.parameters('action', torch.empty(7, device=self.device))
        auxiliary_device = 'cpu' if model == 'hamlet' else self.device
        self.dummy_action = torch.zeros(1, self.recipe['horizon'], 7, device=auxiliary_device)
        self.dummy_pad = torch.zeros(1, self.recipe['horizon'], device=auxiliary_device, dtype=torch.bool)
        self.history = FrameHistory(self.recipe['image_delta_indices'])
        # The same pure-Python helper ships with the robot runtime and cloud checks.
        try:
            from franka_lerobot_recorder.inference_cache import CachedTokenizer, FramePreprocessor, VisualCache
        except ModuleNotFoundError as error:
            if error.name != 'franka_lerobot_recorder':
                raise
            from .contract import FR3
            local = FR3 / 'ros2/src/franka_lerobot_recorder'
            sys.path.insert(0, str(local if local.is_dir() else root / 'deployment_runtime'))
            from franka_lerobot_recorder.inference_cache import CachedTokenizer, FramePreprocessor, VisualCache
        self.visual_cache = VisualCache(self.model)
        if model == "hamlet" and os.environ.get("FR3_HAMLET_VISION_CACHE") == "1":
            from franka_lerobot_recorder.inference_cache import HistoryFeatureCache
            self.visual_cache = HistoryFeatureCache(self.model)
            self.visual_cache.wrap(self.model.model.backbone.model)
        agent.metadata['inference_cache'] = ('hamlet-exact-history-pixels-v1'
            if type(self.visual_cache).__name__ == 'HistoryFeatureCache' else 'immutable-vision-and-exact-text-v1')
        self.image_cache_capacity = agent.hamlet_image_cache_size if model == 'hamlet' else 48
        self.image_preprocessor = FramePreprocessor(lambda frame: self.data.resize_pad(
            torch.from_numpy(frame.copy()).permute(2, 0, 1).float()/255, self.recipe['image_size']),
            capacity=max(1, self.image_cache_capacity))
        if model == 'hamlet':
            from .hamlet_inference import ChatTemplateCache
            processor = self.model.model.collator.processor
            processor.apply_chat_template = ChatTemplateCache(processor)
        if hasattr(self.model, 'tokenizer'):
            self.model.tokenizer = CachedTokenizer(self.model.tokenizer)
        if model == 'xvla':
            self.visual_cache.wrap(self.model.policy.model.vlm, 'get_image_features')
        elif model == 'flower':
            self.visual_cache.wrap(self.model.model.vlm, '_encode_image')
            self.model.model.tokenizer = CachedTokenizer(self.model.model.tokenizer)
        elif model == 'rdt':
            self.visual_cache.wrap(self.model.vision)
        elif model == 'pi05':
            self.visual_cache.wrap(self.model.core.paligemma_with_expert, 'embed_image')
        else:
            self.model.model.collator.processor.tokenizer = CachedTokenizer(self.model.model.collator.processor.tokenizer)

    def warmup(self, cameras):
        """Finish lazy model/graph initialization before starting live camera clocks."""
        images = tuple(np.zeros((cameras[name]['height'], cameras[name]['width'], 3),
                                dtype=np.uint8) for name in CAMERAS)
        state = np.array([0., 0., 0., -1.5, 0., 1.5, 0.], dtype=np.float32)
        self.history.observe(images, 0.)
        try:
            for _ in range(2):
                self.predict(state, images, None)
        finally:
            self.reset()

    def reset(self):
        self.history.clear()
        self.clear_visual_cache()

    def clear_visual_cache(self):
        self.visual_cache.clear()
        self.image_preprocessor.clear()

    def prepare(self, state, images):
        import torch
        frames = self.history.select(images)
        indices = [CAMERAS.index(name) for name in ORDER]
        self.current_frames = [frame[i] for frame in frames for i in indices]
        # No cache is used for caller-owned writable arrays.
        transform = getattr(self, 'image_preprocessor', None)
        if getattr(self, 'image_cache_capacity', 48) == 0:
            transform = None
        image_tensor = torch.stack([torch.stack([
            transform(frame[i]) if transform else self.data.resize_pad(
                torch.from_numpy(frame[i].copy()).permute(2, 0, 1).float()/255, self.recipe['image_size'])
            for i in indices]) for frame in frames])
        horizon = self.recipe['horizon']
        raw = torch.as_tensor(state, dtype=torch.float32)
        normalized_state = ((raw-self.state_center)/self.state_scale if hasattr(self, 'state_center')
                            else self.norm.apply(raw, 'observation.state'))
        # Some training collators require action fields; sampling never receives a target.
        auxiliary_device = 'cpu' if self.recipe['model'] == 'hamlet' else self.device
        return dict(state=normalized_state[None].to(auxiliary_device),
                    # HAMLET's saved collator consumes CPU/PIL images before model.prepare_input uploads them.
                    images=(image_tensor[None] if self.recipe['model'] == 'hamlet'
                            else image_tensor[None].to(self.device)), task=[self.task],
                    action=self.dummy_action if hasattr(self, 'dummy_action') else torch.zeros(1,horizon,7,device=auxiliary_device),
                    action_is_pad=self.dummy_pad if hasattr(self, 'dummy_pad') else torch.zeros(1,horizon,device=auxiliary_device,dtype=torch.bool))

    def predict(self, state, images, check):
        import torch
        batch = self.prepare(state, images)
        if check:
            check()
        with self.visual_cache.observation(self.current_frames), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
            normalized = self.model.predict(batch)
        return (normalized[0].float()*self.action_scale+self.action_center).cpu().numpy()


def run_with_history(executor, options, agent, config):
    if getattr(agent, 'model_name', None) == 'hamlet':
        print('HAMLET: warming inference before starting cameras', flush=True)
        agent.backend.warmup(config['cameras'])
    original = executor.camera_observation
    pump = None

    def observe(cameras, live_config):
        nonlocal pump

        def capture():
            images, stamp = original(cameras, live_config)
            agent.backend.history.observe(images, stamp)
            return images, stamp

        if pump is None:
            # Keep collecting real history through slow model warmup and bridge startup.
            images, stamp = capture()
            pump = executor.ObservationPump(capture, 30)
            agent.backend.history.select(images)
            return images, stamp

        # Only admit complete histories. Wait on the producer, never on control.
        # This is shorter than the worker refresh deadline (100 ms); age guards
        # still apply to the previous prediction throughout the wait.
        deadline = time.monotonic() + .075
        while True:
            images, stamp = pump.refresh()
            try:
                agent.backend.history.select(images)
                return images, stamp
            except HistoryNotReady:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.003)
    executor.camera_observation = observe
    try:
        return executor.run(options, agent, config)
    finally:
        executor.camera_observation = original
        if pump is not None:
            pump.close()
