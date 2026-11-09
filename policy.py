"""Dataset-free inference using the checkpoint's unchanged training adapter."""
from pathlib import Path
import json

import numpy as np
import torch
from safetensors.torch import load_file

from fr3_vla.cloud import load_sources

ROOT = Path(__file__).resolve().parent
TASK = 'hit the red hockey puck until it crosses the center line of the table'
CAMERA_ORDER = ('overhead', 'wrist', 'third')
HISTORY_OFFSETS = (-48, -32, -16, 0)


def prepare_observation(data, normalizer, state, images, task, device):
    state = np.asarray(state, dtype=np.float32)
    images = np.asarray(images)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('state must be seven finite absolute joint angles in radians')
    if (images.ndim != 5 or images.shape[:2] != (4, 3) or images.shape[-1] != 3
            or min(images.shape[2:4]) < 1 or images.dtype != np.uint8):
        raise ValueError('images must be RGB uint8 [4 history, 3 cameras, height, width, 3]')
    if not isinstance(task, str) or not task.strip():
        raise ValueError('task must be a nonempty string')
    frames = torch.from_numpy(images.copy()).permute(0, 1, 4, 2, 3).float() / 255
    frames = data.resize_pad(frames.flatten(0, 1), 224).reshape(4, 3, 3, 224, 224)
    return {
        'state': normalizer.apply(torch.from_numpy(state.copy()), 'observation.state')[None].to(device),
        'images': frames[None].to(device),
        'action': torch.zeros(1, 50, 7, device=device),
        'action_is_pad': torch.zeros(1, 50, dtype=torch.bool, device=device),
        'task': [task],
    }


class HamletPolicy:
    """One call consumes the complete four-frame history and returns [50, 7] radians.

    Images are ordered oldest to newest, then overhead/wrist/third. Camera data
    must come from the same episode at 30 Hz offsets [-48, -32, -16, 0]. The caller
    repeats the episode's first frame only for history before the episode start.
    Instances mutate action-head memory and must not be shared across threads.
    """
    def __init__(self, checkpoint=ROOT, training_root=ROOT, device='cuda'):
        checkpoint, training_root = Path(checkpoint).resolve(), Path(training_root).resolve()
        config = json.loads((checkpoint / 'config.json').read_text())
        contract = json.loads((checkpoint / 'training_contract.json').read_text())
        if (config['model'] != 'hamlet' or contract['image_size'] != 224
                or contract['image_delta_indices'] != list(HISTORY_OFFSETS)
                or contract['camera_order'] != list(CAMERA_ORDER)
                or contract['normalization'] != 'bounds' or contract['horizon'] != 50):
            raise ValueError('This loader supports the v6 224x224, 20k HAMLET recipe')
        if not (checkpoint / 'COMPLETE').is_file():
            raise ValueError('Checkpoint has no COMPLETE marker')
        self.device = torch.device(device)
        self.data, adapters = load_sources(checkpoint, training_root)
        audit = json.loads((checkpoint / 'data_audit.json').read_text())
        self.normalizer = self.data.Normalizer(audit['normalization'], 'bounds')
        # The adapter constructs HAMLET, loads frozen GR00T weights, and sets the
        # exact training configuration before the trained tensors are overlaid.
        self.model = adapters.HamletAdapter()
        state = load_file(str(checkpoint / 'trainable.safetensors'))
        parameters = dict(self.model.named_parameters())
        expected = {k for k, v in parameters.items() if v.requires_grad}
        if set(state) != expected:
            raise ValueError('Trainable tensor keys differ from the training adapter')
        uncovered = {f'model.{k}' for k in self.model.load_report['missing']} - set(state)
        if uncovered:
            raise ValueError(f'Uninitialized HAMLET tensors: {sorted(uncovered)}')
        with torch.no_grad():
            for name, value in state.items():
                if parameters[name].shape != value.shape or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid trained tensor: {name}')
                parameters[name].copy_(value)
        del state
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, state, images, task=TASK, seed=None):
        batch = prepare_observation(self.data, self.normalizer, state, images, task, self.device)
        # fork_rng preserves the caller's generator state when a seed is supplied.
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
            # Both devices need autocast at the BF16 backbone / FP32 head boundary.
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                normalized = self.model.predict(batch)[0].float()
        actions = self.normalizer.undo(normalized).cpu().numpy()
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError('Model returned invalid actions')
        return actions
