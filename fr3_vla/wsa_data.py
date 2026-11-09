"""FR3 mappings for WSA's LeRobot fork; the v4 dataset itself stays unchanged."""
import numpy as np

ROBOT_TYPE = 'fr3_teach_aligned_single_arm'
WSA_CAMERAS = ('overhead', 'wrist', 'third')
IMAGE_OFFSETS = (-15, 0, 15)


def register_v4():
    import torch
    from lerobot.transforms.constants import FEATURE_MAPPING, IMAGE_MAPPING, MASK_MAPPING
    FEATURE_MAPPING[ROBOT_TYPE] = {'observation.state': ['observation.state'], 'action': ['action']}
    IMAGE_MAPPING[ROBOT_TYPE] = {f'observation.images.{name}': f'observation.images.image{i}'
                               for i, name in enumerate(WSA_CAMERAS)}
    MASK_MAPPING[ROBOT_TYPE] = torch.ones(7, dtype=torch.bool)


def training_sample(dataset, episode, frame, horizon):
    """Raw layout preview: only training includes the future image supervision target."""
    table = dataset.table(episode)
    if not 0 <= frame < len(table) or horizon < 1:
        raise ValueError('Invalid WSA sample frame/horizon')
    image_indices = np.clip(np.array(IMAGE_OFFSETS) + frame, 0, len(table) - 1)
    action_indices = np.minimum(np.arange(frame, frame + horizon), len(table) - 1)
    return {'observation.state': np.asarray(table.iloc[frame]['observation.state'], dtype=np.float32),
            'action': np.stack(table.action).astype(np.float32)[action_indices],
            'action_is_pad': np.arange(frame, frame + horizon) >= len(table),
            'task': dataset.task,
            **{f'observation.images.image{i}': np.stack([dataset.image(episode, int(t), name)
                for t in image_indices]) for i, name in enumerate(WSA_CAMERAS)}}
