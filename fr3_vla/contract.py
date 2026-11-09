"""The shared training/deployment contract, independent of any model framework."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path('/home/user1/lerobot_data/hockey_three_cameras_aligned_v4')
FR3 = Path('/home/user1/gello_software')
CAMERAS = ('wrist', 'overhead', 'third')
JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]
MODELS = ('xvla', 'flower', 'hamlet', 'rdt', 'pi05', 'wsa')
LOWER = np.array([-2.8007, -1.7361, -2.8007, -2.9770, -2.7763, .5398, -2.9508])
UPPER = np.array([2.8007, 1.7361, 2.8007, -.2169, 2.7763, 4.5216, 2.9508])


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_dataset_contract(root):
    root = Path(root).expanduser().resolve(strict=True)
    contract = json.loads((root / 'recording/contract.json').read_text())
    info = json.loads((root / 'meta/info.json').read_text())
    if (info['codebase_version'] != 'v3.0' or info['fps'] != 30 or contract['fps'] != 30
            or contract['units']['arm_position'] != 'rad'
            or contract['action_definition'] != 'next_uniform_time_interpolated_joint_position_rad'
            or not np.isclose(contract['action_horizon_s'], 1 / 30)
            or contract.get('gripper_joint_names') != []
            or set(contract['cameras']) != set(CAMERAS)):
        raise ValueError('Expected v4: 30 Hz, aligned next-frame absolute joint radians, three cameras, fixed gripper')
    for key in ('observation.state', 'action'):
        for feature in (contract['features'][key], info['features'][key]):
            if feature['shape'] != [7] or feature['names'] != JOINTS:
                raise ValueError(f'{key}: expected ordered FR3 J1-J7')
    for name in CAMERAS:
        camera = contract['cameras'][name]
        if camera['fps'] != 30 or camera['rotate_180'] is not False:
            raise ValueError(f'Unexpected camera timing/orientation: {name}')
        expected = [camera['height'], camera['width'], 3]
        if info['features'][f'observation.images.{name}']['shape'] != expected:
            raise ValueError(f'Camera resolution differs from contract: {name}')
    return contract, info


def validate_observation(state, images, cameras):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('State must be seven finite joint radians')
    if len(images) != 3:
        raise ValueError('All three RGB camera frames are required')
    for name, frame in zip(CAMERAS, images, strict=True):
        camera = cameras[name]
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.shape != (camera['height'], camera['width'], 3)):
            raise ValueError(f'{name}: expected RGB uint8 HWC at the recorded resolution')
    return state


def validate_actions(actions, horizon=None):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
        raise ValueError('Expected a nonempty [horizon, 7] action chunk')
    if horizon is not None and len(actions) != horizon:
        raise ValueError('Action horizon differs from the exported checkpoint')
    if not np.isfinite(actions).all() or (actions < LOWER).any() or (actions > UPPER).any():
        raise ValueError('Predicted absolute radians exceed finite FR3 joint limits')
    return actions


def normalize(values, stats):
    low, high = np.asarray(stats['min']), np.asarray(stats['max'])
    width = high - low
    return np.where(width > 1e-8, 2 * (values - low) / np.maximum(width, 1e-8) - 1, 0).astype(np.float32)


def denormalize(values, stats):
    low, high = np.asarray(stats['min']), np.asarray(stats['max'])
    return ((values + 1) * (high - low) / 2 + low).astype(np.float32)


def pad_xvla(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 7:
        raise ValueError('X-VLA FR3 input must have seven joints')
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 13)])


def pack_rdt(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 7:
        raise ValueError('RDT FR3 input must have seven joints')
    result = np.zeros((*values.shape[:-1], 128), dtype=np.float32)
    result[..., :7] = values
    mask = np.zeros(128, dtype=np.float32)
    mask[:7] = 1
    return result, mask


def hamlet_observation(state, images, task):
    return dict(video={name: frame[None, None] for name, frame in zip(CAMERAS, images, strict=True)},
                state={'joint_position': np.asarray(state, dtype=np.float32)[None, None]},
                language={'annotation.human.action.task_description': [[task]]})


def read_manifest(checkpoint, model, dataset_root, *, verify_weights=True, training_root=None):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    manifest = json.loads((checkpoint / 'fr3_deployment.json').read_text())
    if training_root is not None:
        from .cloud import FORMAT
        if manifest.get('checkpoint_format') != FORMAT:
            raise ValueError('--training-root requires a vla-v4 trainable checkpoint export')
        manifest['assets']['training_root'] = str(Path(training_root).expanduser().resolve(strict=True))
    contract, _ = read_dataset_contract(dataset_root)
    if (manifest.get('schema_version') != 1 or manifest.get('model') != model
            or manifest.get('joint_names') != JOINTS or manifest.get('camera_order') != list(CAMERAS)
            or manifest.get('fps') != 30 or manifest.get('action_units') != 'rad'
            or manifest.get('action_definition') != contract['action_definition']
            or manifest.get('cameras') != contract['cameras']
            or manifest.get('contract_sha256') != digest(Path(dataset_root) / 'recording/contract.json')):
        raise ValueError('Checkpoint deployment manifest does not match the v4 robot/data contract')
    horizon = manifest.get('horizon')
    replan = manifest.get('replan_steps')
    if type(horizon) is not int or type(replan) is not int or not 1 <= replan <= horizon:
        raise ValueError('Invalid checkpoint horizon/replan interval')
    if replan / 30 >= .5:
        raise ValueError('Replan interval exceeds the executor freshness budget')
    if not manifest.get('weights_sha256'):
        raise ValueError('Unsealed template: export the trained checkpoint with scripts/seal_checkpoint.py')
    for relative, expected in manifest['weights_sha256'].items():
        path = (checkpoint / relative).resolve(strict=True)
        if not path.is_relative_to(checkpoint):
            raise ValueError('Checkpoint asset escapes its directory')
        if verify_weights and digest(path) != expected:
            raise ValueError(f'Checkpoint asset hash mismatch: {relative}')
    for name, files in manifest.get('assets_sha256', {}).items():
        base = Path(manifest['assets'][name]).expanduser().resolve(strict=True)
        for relative, expected in files.items():
            path = (base / relative).resolve(strict=True)
            if not path.is_relative_to(base) or (verify_weights and digest(path) != expected):
                raise ValueError(f'Base asset hash/path mismatch: {name}/{relative}')
    if model == 'wsa':
        from .wsa import validate_export
        validate_export(checkpoint, manifest)
    if manifest.get('checkpoint_format'):
        from .cloud import FORMAT, validate_export
        if manifest['checkpoint_format'] != FORMAT:
            raise ValueError('Unknown deployment checkpoint format')
        validate_export(checkpoint, manifest)
    return manifest


def validate_live_config(manifest, config):
    if (config.get('collection_mode') != 'teach' or config['arm_joint_names'] != JOINTS
            or config['dataset']['fps'] != 30):
        raise ValueError('Live configuration must use FR3 Teach with J1-J7 at 30 Hz')
    for name in CAMERAS:
        recorded, live = manifest['cameras'][name], config['cameras'][name]
        for field in ('serial', 'model', 'width', 'height', 'fps', 'rotate_180'):
            if live[field] != recorded[field]:
                raise ValueError(f'Live camera differs from training: {name}.{field}')
