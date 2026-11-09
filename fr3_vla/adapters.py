"""Model-specific local inference. One model per process avoids upstream package-name collisions."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import numpy as np

from .contract import (CAMERAS, ROOT, denormalize, hamlet_observation, normalize,
                       pad_xvla, read_manifest, validate_actions, validate_observation)


def add_upstream(model):
    path = ROOT / f'models/{model}/upstream'
    if not path.is_dir():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path / 'src' if model == 'wsa' else path))


def strict_hf_load(cls, path, **kwargs):
    model, info = cls.from_pretrained(path, local_files_only=True, output_loading_info=True, **kwargs)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Incomplete checkpoint load: {info}')
    return model


class Agent:
    def __init__(self, model, checkpoint, dataset_root, device='cuda', training_root=None, *, hamlet_kv_cache=None,
                 hamlet_bf16_linear=None, hamlet_image_cache_size=None, hamlet_cuda_graph=None):
        import torch
        if hamlet_kv_cache is None:
            hamlet_kv_cache = model == 'hamlet'
        if hamlet_kv_cache and model != 'hamlet':
            raise ValueError('Action-head KV caching is supported only for HAMLET')
        if model != 'hamlet' and (hamlet_bf16_linear or hamlet_cuda_graph or hamlet_image_cache_size is not None):
            raise ValueError('HAMLET optimization options require model hamlet')
        self.hamlet_bf16_linear = model == 'hamlet' if hamlet_bf16_linear is None else hamlet_bf16_linear
        self.hamlet_cuda_graph = model == 'hamlet' if hamlet_cuda_graph is None else hamlet_cuda_graph
        self.hamlet_image_cache_size = 96 if hamlet_image_cache_size is None else hamlet_image_cache_size
        if type(self.hamlet_image_cache_size) is not int or self.hamlet_image_cache_size < 0:
            raise ValueError('HAMLET image cache size must be a nonnegative integer')
        self.model_name = model
        self.checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        self.manifest = read_manifest(self.checkpoint, model, dataset_root, training_root=training_root)
        self.device = device
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this session')
        self.metadata = dict(self.manifest, dataset_root=str(Path(dataset_root).resolve()))
        self.camera_keys = tuple(f'observation.images.{name}' for name in CAMERAS)
        self.policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=self.manifest['replan_steps'],
                                                             chunk_size=self.manifest['horizon']))
        from .cloud import FORMAT, CloudBackend
        self.cloud_runtime = self.manifest.get('checkpoint_format') == FORMAT
        self.min_refresh_steps = 4 if model == 'hamlet' and not self.cloud_runtime else 1
        self.backend = CloudBackend(self) if self.cloud_runtime else getattr(self, f'_load_{model}')()
        if model == 'hamlet':
            from .hamlet_inference import configure_action_head
            head = self.backend.model.model.action_head if self.cloud_runtime else self.backend.model.action_head
            head.use_encoder_kv_cache = hamlet_kv_cache
            bf16 = self.hamlet_bf16_linear and device.startswith('cuda')
            graph = self.hamlet_cuda_graph and device.startswith('cuda')
            self.action_graph = configure_action_head(head, bf16_linear=bf16, cuda_graph=graph)
            self.metadata['hamlet_kv_cache'] = hamlet_kv_cache
            self.metadata.update(hamlet_bf16_linear=bf16, hamlet_image_cache_size=self.hamlet_image_cache_size,
                                 hamlet_cuda_graph=graph)
            self.metadata['inference_cache'] = (
                self.metadata.get('inference_cache', 'legacy-hamlet')
                + f';action-head-kv={"on" if hamlet_kv_cache else "off"}'
                + f';bf16-linear={bf16};image-cache={self.hamlet_image_cache_size};cuda-graph={graph}')
        self.reset()

    def asset(self, key):
        relative = self.manifest['assets'][key]
        path = Path(relative).expanduser()
        path = path if path.is_absolute() else self.checkpoint / path
        return path.resolve(strict=True)

    def reset(self):
        self.session_id = uuid.uuid4().hex
        if getattr(self, 'cloud_runtime', False):
            self.backend.reset()
        elif self.model_name == 'hamlet':
            # Upstream reset() is a no-op; clear both session caches explicitly.
            self.backend._memory_cache.clear()
            self.backend._vision_session_cache.clear()
            self.backend.model.action_head._memory_cache = None
            self.backend.model.action_head._vision_cache = None
        elif hasattr(self.backend, 'reset'):
            self.backend.reset()

    def clear_visual_cache(self):
        if getattr(self, 'cloud_runtime', False):
            self.backend.clear_visual_cache()

    def predict_chunk(self, state, wrist, overhead, third, *, freshness_check=None):
        import torch
        images = (wrist, overhead, third)
        state = validate_observation(state, images, self.manifest['cameras'])
        with torch.inference_mode():
            if getattr(self, 'cloud_runtime', False):
                result = self.backend.predict(state, images, freshness_check)
            else:
                result = getattr(self, f'_predict_{self.model_name}')(state, images, freshness_check)
        return validate_actions(result, self.manifest['horizon'])

    def _load_xvla(self):
        add_upstream('xvla')
        from models.modeling_xvla import XVLA
        from models.processing_xvla import XVLAProcessor
        model = strict_hf_load(XVLA, self.checkpoint)
        c = model.config
        if (c.action_mode != 'auto' or c.real_action_dim != 7 or c.max_action_dim != 20
                or c.num_actions != self.manifest['horizon'] or c.num_domains <= 29):
            raise ValueError('Train/export X-VLA with auto/7/20, domain 29 and the matching horizon')
        self.processor = XVLAProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        return model.to(self.device).eval()

    def _load_wsa(self):
        import subprocess
        subprocess.run([sys.executable, str(ROOT / 'scripts/patch_wsa_transformers.py'), '--check'], check=True)
        add_upstream('wsa')
        from .wsa import WSABackend
        return WSABackend(self)

    def _predict_wsa(self, state, images, check):
        return self.backend.predict(state, images, check)

    def _predict_xvla(self, state, images, check):
        import torch
        from PIL import Image
        inputs = self.processor([Image.fromarray(frame) for frame in images], self.manifest['task'])
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        inputs['proprio'] = torch.from_numpy(pad_xvla(state)[None]).to(self.device)
        inputs['domain_id'] = torch.tensor([29], device=self.device)
        if check:
            check()
        return self.backend.generate_actions(**inputs, steps=self.manifest['inference']['steps'])[0].float().cpu().numpy()

    def _load_flower(self):
        import torch
        add_upstream('flower')
        from .flower_model import FR3FLOWER
        options = json.loads((self.checkpoint / 'fr3_flower_config.json').read_text())
        options['vlm_path'] = str(self.asset('florence'))
        options['load_pretrained'] = False
        options['pretrained_model_path'] = None
        if (options.get('action_dim') != 8 or options.get('lowdim_obs_dim') != 8
                or options.get('use_proprio') is not True
                or options.get('act_window_size') != self.manifest['horizon']):
            raise ValueError('FLOWER checkpoint config differs from FR3 training architecture')
        model = FR3FLOWER(**options)
        from safetensors.torch import load_file
        model.load_state_dict(load_file(self.checkpoint / 'model.safetensors'), strict=True)
        return model.to(self.device).eval()

    def _predict_flower(self, state, images, check):
        import torch
        from .data import flower_image
        batch = {'rgb_obs': {}, 'lang_text': [self.manifest['task']]}
        for name, image in zip(CAMERAS, images, strict=True):
            batch['rgb_obs'][name] = torch.from_numpy(flower_image(image)[None, None]).to(self.device)
        proprio = np.pad(normalize(state, self.manifest['statistics']['observation.state']), (0, 1))
        batch['proprio'] = torch.from_numpy(proprio[None]).to(self.device)
        if check:
            check()
        normalized = self.backend(batch)[0, :, :7].float().cpu().numpy()
        return denormalize(normalized, self.manifest['statistics']['action'])

    def _load_hamlet(self):
        add_upstream('hamlet')
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from .hamlet_config import fr3_v4_joint
        backend = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, str(self.checkpoint), device=self.device, strict=True)
        expected = {'video': list(CAMERAS), 'state': ['joint_position'], 'action': ['joint_position'],
                    'language': ['annotation.human.action.task_description']}
        for key, names in expected.items():
            if backend.modality_configs[key].modality_keys != names:
                raise ValueError(f'HAMLET checkpoint has wrong FR3 {key} modalities')
        if len(backend.modality_configs['action'].delta_indices) != self.manifest['horizon']:
            raise ValueError('HAMLET action horizon mismatch')
        config = backend.model.config
        if (getattr(config, 'hamlet_mode', None) != 'finetune'
                or getattr(config, 'memory_window', None) != 4
                or getattr(config, 'memory_stride', None) != 4):
            raise ValueError('Expected HAMLET finetune with window=4, stride=4')
        from safetensors import safe_open
        saved = set()
        for file in sorted(self.checkpoint.glob('model*.safetensors')):
            with safe_open(file, framework='pt', device='cpu') as weights:
                saved.update(weights.keys())
        state = backend.model.state_dict()
        aliases = {}
        for name, parameter in backend.model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        covered = set(saved)
        for names in aliases.values():
            if saved.intersection(names):
                covered.update(names)
        if set(state) - covered or saved - set(state):
            raise ValueError('HAMLET checkpoint does not cover the complete model state')
        return backend

    def _predict_hamlet(self, state, images, check):
        observation = hamlet_observation(state, images, self.manifest['task'])
        if check:
            check()
        actions, _ = self.backend.get_action(observation, options={'session_ids': [self.session_id]})
        return actions['joint_position'][0]

    def _load_rdt(self):
        import torch
        import yaml
        add_upstream('rdt')
        from scripts.agilex_model import RoboticDiffusionTransformerModel
        from models.rdt_runner import RDTRunner

        class FR3RDT(RoboticDiffusionTransformerModel):
            def get_policy(inner, pretrained):
                # Upstream's default loading uses strict=False; deployment requires strict weights.
                return RDTRunner.from_pretrained(str(pretrained), strict=True, local_files_only=True)

            def _format_joint_to_state(inner, joints):
                if joints.shape[-1] != 7:
                    raise ValueError('Expected FR3 J1-J7')
                state = joints.new_zeros((*joints.shape[:-1], 128))
                state[..., :7] = joints
                mask = joints.new_zeros((joints.shape[0], 128))
                mask[:, :7] = 1
                return state, mask

            def _unformat_action_to_joint(inner, actions):
                return actions[..., :7]

        config = yaml.safe_load((self.checkpoint / 'rdt_config.yaml').read_text())
        if (config['common']['state_dim'] != 128 or config['common']['num_cameras'] != 3
                or config['common']['img_history_size'] != 2
                or config['common']['action_chunk_size'] != self.manifest['horizon']):
            raise ValueError('RDT checkpoint has an incompatible state/image/action layout')
        language = torch.load(self.checkpoint / 'language.pt', map_location='cpu', weights_only=True)
        if not isinstance(language, dict) or language.get('instruction') != self.manifest['task']:
            raise ValueError('RDT language.pt must include the exact original instruction and its T5 embeddings')
        self.text_embedding = language['embeddings']
        if (self.text_embedding.ndim != 3 or self.text_embedding.shape[0] != 1
                or self.text_embedding.shape[-1] != config['model']['lang_token_dim']
                or not torch.isfinite(self.text_embedding).all()):
            raise ValueError('RDT language.pt must contain the finite T5 embedding [1, L, D] of the manifest task')
        return FR3RDT(config, device=self.device, dtype=torch.bfloat16 if self.device.startswith('cuda') else torch.float32,
                      control_frequency=30, pretrained=str(self.checkpoint),
                      pretrained_vision_encoder_name_or_path=str(self.asset('siglip')))

    def _predict_rdt(self, state, images, check):
        import torch
        from PIL import Image
        ordered = [Image.fromarray(images[i]) for i in (1, 0, 2)]
        if check:
            check()
        return self.backend.step(torch.from_numpy(state[None]), ordered + ordered, self.text_embedding)[0].cpu().numpy()

    def _load_pi05(self):
        project = Path(self.manifest['inference']['project']).resolve(strict=True)
        spec = importlib.util.spec_from_file_location('fr3_v4_pi05', project / 'inference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backend = module.HockeyPI05(self.checkpoint, device=self.device,
                        dataset_root=self.metadata['dataset_root'], task=self.manifest['task'])
        if (backend.camera_keys != self.camera_keys or backend.policy.config.chunk_size != self.manifest['horizon']):
            raise ValueError('PI0.5 checkpoint does not contain the v4 three-camera/horizon configuration')
        return backend

    def _predict_pi05(self, state, images, check):
        return self.backend.predict_chunk(state, *images, freshness_check=check)
