"""WSA-Base local inference using the official preprocessing and the shared FR3 executor."""
from collections import deque
import json
import math
import os
from threading import Lock

import numpy as np

from .contract import CAMERAS
from .wsa_data import IMAGE_OFFSETS, ROBOT_TYPE, WSA_CAMERAS, register_v4


def validate_export(checkpoint, manifest):
    train = json.loads((checkpoint / 'train_config.json').read_text())
    config = json.loads((checkpoint / 'config.json').read_text())
    inference = manifest['inference']
    if (train.get('dataset', {}).get('action_mode') != 'abs'
            or inference.get('action_mode') != 'abs'
            or inference.get('normalization') != 'mean_std'
            or inference.get('camera_order') != list(WSA_CAMERAS)
            or inference.get('image_delta_indices') != list(IMAGE_OFFSETS)
            or type(inference.get('steps')) is not int or not 1 <= inference['steps'] <= 100):
        raise ValueError('WSA requires the FR3 abs/mean_std/three-view/history training recipe')
    if (config.get('type') not in ('WSA_Base', 'wsa_base')
            or config.get('chunk_size') != manifest['horizon']
            or config.get('image_delta_indices') != list(IMAGE_OFFSETS)
            or config.get('input_features', {}).get('observation.state', {}).get('shape') != [7]
            or config.get('output_features', {}).get('action', {}).get('shape') != [7]
            or config.get('max_state_dim', 32) != 32 or config.get('max_action_dim', 32) != 32
            or config.get('image_resolution') != [224, 224]):
        raise ValueError('WSA-Base checkpoint features must match FR3: seven joints, 32 padded dimensions, 224 RGB')
    stats = json.loads((checkpoint / 'stats.json').read_text())[ROBOT_TYPE]
    for key in ('observation.state', 'action'):
        for field in ('mean', 'std', 'min', 'max'):
            actual = np.asarray(stats[key][field], dtype=np.float32)
            expected = np.asarray(manifest['statistics'][key][field], dtype=np.float32)
            if (actual.shape != (7,) or expected.shape != (7,) or not np.isfinite(actual).all()
                    or not np.array_equal(actual, expected) or (field == 'std' and (actual < 0).any())):
                raise ValueError(f'WSA training/deployment statistics differ: {key}.{field}')
    return stats


class FrameHistory:
    """Keep source snapshots at observation cadence, independent of inference latency."""
    def __init__(self):
        self.lock = Lock()
        self.frames = deque(maxlen=90)

    def clear(self):
        with self.lock:
            self.frames.clear()

    def observe(self, images, stamp):
        if len(images) != 3 or not math.isfinite(stamp):
            raise ValueError('WSA history needs three cameras and a finite monotonic timestamp')
        with self.lock:
            if self.frames:
                if stamp < self.frames[-1][0]:
                    raise RuntimeError('WSA camera timestamps regressed')
                if stamp == self.frames[-1][0] and all(a is b for a, b in zip(self.frames[-1][1], images, strict=True)):
                    return
            self.frames.append((stamp, tuple(images)))

    def pair(self, current):
        with self.lock:
            snapshots = list(self.frames)
        anchor = next((i for i in range(len(snapshots) - 1, -1, -1)
                       if all(a is b for a, b in zip(snapshots[i][1], current, strict=True))), None)
        if anchor is None:
            raise RuntimeError('WSA current frame is absent from the observation history')
        stamp = snapshots[anchor][0]
        target = stamp + IMAGE_OFFSETS[0] / 30
        candidates = [entry for entry in snapshots[:anchor + 1] if entry[0] <= target + 1e-6]
        past_stamp, past = candidates[-1] if candidates else snapshots[0]
        # Episode-start padding repeats the earliest frame, just like LeRobot sampling.
        if candidates and target - past_stamp > 2 / 30:
            raise RuntimeError('WSA history has a camera gap near the required past frame')
        return past, tuple(current)


class WSABackend:
    def __init__(self, agent):
        os.environ['HF_HUB_OFFLINE'] = '1'
        import torch
        from safetensors.torch import load_model
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.WSA_Base.modeling_wsa_base import WSABasePolicy
        from lerobot.policies.WSA_Base.transform_wsa_base import Qwen3_VLProcessorTransformFn
        from lerobot.transforms.core import NormalizeTransformFn, ResizeImagesWithPadFn, UnNormalizeTransformFn
        register_v4()
        stats = validate_export(agent.checkpoint, agent.manifest)
        self.history = FrameHistory()
        self.device = agent.device
        self.dtype = torch.bfloat16 if self.device.startswith('cuda') else torch.float32
        config = PreTrainedConfig.from_pretrained(agent.checkpoint, local_files_only=True)
        config.device = self.device
        config.dtype = 'bfloat16' if self.dtype == torch.bfloat16 else 'float32'
        config.compile_model = False
        config.lambda_3d = 0.0  # Match the official server: keep learned 3D queries, skip the training teacher.
        config.qwen3_vl_pretrained_path = str(agent.asset('qwen'))
        config.cosmos_tokenizer_path_or_name = str(agent.asset('cosmos'))
        config.n_action_steps = agent.manifest['replan_steps']
        config.num_inference_steps = agent.manifest['inference']['steps']
        if not all((agent.asset('cosmos') / name).is_file() for name in ('encoder.jit', 'decoder.jit')):
            raise FileNotFoundError('The local Cosmos asset requires encoder.jit and decoder.jit')
        self.policy = WSABasePolicy(config)
        missing, unexpected = load_model(self.policy, str(agent.checkpoint / 'model.safetensors'),
                                         strict=False, device=self.device)
        # Only frozen external assets are excluded from an official complete WSA export.
        missing = [key for key in missing if not key.startswith(('model.cosmos.', 'model.da3_teacher.'))]
        if missing or unexpected:
            raise ValueError(f'Incomplete WSA checkpoint load: missing={missing}, unexpected={unexpected}')
        self.policy.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        self.resize = ResizeImagesWithPadFn(height=224, width=224)
        norm_stats = {key: {field: np.asarray(value, dtype=np.float32) for field, value in stats[key].items()
                           if field in ('mean', 'std', 'min', 'max')}
                      for key in ('observation.state', 'action')}
        self.normalize = NormalizeTransformFn(selected_keys=['observation.state'], mode='mean_std', norm_stats=norm_stats)
        self.unnormalize = UnNormalizeTransformFn(selected_keys=['action'], mode='mean_std', norm_stats=norm_stats)
        self.processor = Qwen3_VLProcessorTransformFn(pretrained_model_name_or_path=str(agent.asset('qwen')),
                                 max_length=config.tokenizer_max_length)
        self.task = agent.manifest['task']

    def reset(self):
        self.policy.reset()
        self.history.clear()

    def predict(self, state, images, check):
        import torch
        past, current = self.history.pair(images)
        indices = [CAMERAS.index(name) for name in WSA_CAMERAS]
        sample = {'observation.state': torch.from_numpy(state.copy()), 'task': self.task}
        for i, source in enumerate(indices):
            pair = np.stack((past[source], current[source]))
            sample[f'observation.images.image{i}'] = torch.from_numpy(pair).permute(0, 3, 1, 2).float() / 255
        sample = self.resize(sample)
        for i in range(3):
            sample[f'observation.images.image{i}_mask'] = torch.tensor(True)
        sample = self.normalize(self.processor(sample))
        batch = {key: value[None].to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype)
                 for key, value in sample.items() if isinstance(value, torch.Tensor)}
        if check:
            check()
        actions, _ = self.policy.predict_action_chunk(batch, decode_image=False)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 7:
            raise ValueError(f'WSA returned an invalid action shape: {tuple(actions.shape)}')
        return self.unnormalize({'action': actions[0].float()})['action'].cpu().numpy()


def run_with_history(executor, options, agent, config):
    original = executor.camera_observation

    def observe(cameras, live_config):
        frames, stamp = original(cameras, live_config)
        agent.backend.history.observe(frames, stamp)
        return frames, stamp

    executor.camera_observation = observe
    try:
        return executor.run(options, agent, config)
    finally:
        executor.camera_observation = original
