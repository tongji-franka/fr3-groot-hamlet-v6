"""Configuration shared by ROS Python 3.10 and franka Python 3.11."""
import copy
import math
from pathlib import Path
import sys


def default_config_path():
    relative = Path('config/single_fr3_d435.yaml')
    candidates = [Path(__file__).parent.parent / relative,
                  Path(sys.prefix) / 'share/franka_lerobot_recorder' / relative]
    for parent in Path(__file__).parents:
        candidates.append(parent / 'share/franka_lerobot_recorder' / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Specify --config pointing to single_fr3_d435.yaml')


def load_config(path=None):
    import yaml
    path = Path(path) if path else default_config_path()
    config = yaml.safe_load(path.read_text())
    validate_config(config)
    return copy.deepcopy(config), path.resolve()


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping')
    mode = config.get('collection_mode', 'gello')
    if mode not in ('gello', 'teach', 'teach_gripper'):
        raise ValueError('collection_mode must be gello, teach or teach_gripper')
    if mode in ('teach', 'teach_gripper') and config['dataset']['record_reference']:
        raise ValueError('Teach does not use GELLO controller references')
    bridge = config['bridge']
    if bridge['host'] not in ('127.0.0.1', 'localhost'):
        raise ValueError('Bridge must bind to loopback on the same host')
    if not isinstance(bridge['port'], int) or not 1024 <= bridge['port'] <= 65535:
        raise ValueError('Invalid bridge port')
    if bridge['qos'] not in ('best_effort', 'reliable'):
        raise ValueError('qos must be best_effort or reliable')
    joint_fields = [('arm_joint_names', 7)]
    if mode == 'gello':
        joint_fields.append(('gripper_joint_names', 2))
    for key, count in joint_fields:
        names = config[key]
        if len(names) != count or len(set(names)) != count or not all(
                isinstance(n, str) and n for n in names):
            raise ValueError(f'{key} must contain {count} unique joint names')
    topics = ('state',) if mode == 'teach' else ('state', 'gripper_state') if mode == 'teach_gripper' else (
        'state', 'action', 'gripper_state', 'gripper_command', 'reference')
    for name in topics:
        if not isinstance(config['topics'][name], str) or not config['topics'][name]:
            raise ValueError(f'Invalid {name} topic')
    fps = config['dataset']['fps']
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError('dataset.fps must be a positive integer')
    if not isinstance(config['dataset']['record_reference'], bool):
        raise ValueError('record_reference must be boolean')
    if not isinstance(config['dataset']['writer_threads'], int) or config['dataset']['writer_threads'] <= 0:
        raise ValueError('writer_threads must be a positive integer')
    for value in config['timing'].values():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Timing thresholds must be finite and positive')
    if not isinstance(bridge['publish_hz'], (int, float)) or not math.isfinite(
            bridge['publish_hz']) or bridge['publish_hz'] <= 0:
        raise ValueError('bridge.publish_hz must be finite and positive')
    for key, camera in config.get('cameras', {}).items():
        if not key or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in key):
            raise ValueError('Camera keys must use lowercase letters, digits and underscores')
        for size in ('width', 'height', 'fps'):
            if not isinstance(camera[size], int) or camera[size] <= 0:
                raise ValueError(f'Invalid camera {key} {size}')
        if not isinstance(camera['rotate_180'], bool):
            raise ValueError('rotate_180 must be boolean')
