"""Read LeRobot v3 episodes with their own video offsets; never shift v4 actions again."""
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .contract import CAMERAS, read_dataset_contract


def flower_image(rgb):
    """Legacy full FLOWER exports use CLIP normalization; cloud checkpoints use saved sources."""
    image = ImageOps.pad(Image.fromarray(rgb), (224, 224), method=Image.Resampling.BICUBIC,
                         color=(123, 117, 104))
    array = np.asarray(image, dtype=np.float32) / 255
    mean = np.array([.48145466, .4578275, .40821073], dtype=np.float32)
    std = np.array([.26862954, .26130258, .27577711], dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1).copy()


class V4Dataset:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve(strict=True)
        self.contract, self.info = read_dataset_contract(self.root)
        files = sorted((self.root / 'meta/episodes').rglob('*.parquet'))
        self.episodes = pd.concat([pd.read_parquet(p) for p in files]).sort_values('episode_index')
        self.episodes = self.episodes.set_index('episode_index', drop=False)
        self.tasks = pd.read_parquet(self.root / 'meta/tasks.parquet')
        if len(self.tasks) != 1:
            raise ValueError('This FR3 v4 adapter expects exactly one original task instruction')
        self.task = str(self.tasks.index[0])
        self._table_key = self._table = None

    def table(self, episode_index):
        meta = self.episodes.loc[episode_index]
        key = int(meta['data/chunk_index']), int(meta['data/file_index'])
        if self._table_key != key:
            path = self.info['data_path'].format(chunk_index=key[0], file_index=key[1])
            self._table = pd.read_parquet(self.root / path)
            self._table_key = key
        table = self._table[self._table.episode_index == episode_index].sort_values('frame_index')
        length = int(meta['length'])
        if (len(table) != length or not np.array_equal(table.frame_index, np.arange(length))
                or not np.allclose(table.timestamp, np.arange(length) / 30, atol=1e-4)):
            raise ValueError(f'Episode {episode_index}: frame/timestamp metadata mismatch')
        return table

    def video(self, episode_index, name):
        key = f'observation.images.{name}'
        row = self.episodes.loc[episode_index]
        path = self.info['video_path'].format(video_key=key,
                chunk_index=int(row[f'videos/{key}/chunk_index']),
                file_index=int(row[f'videos/{key}/file_index']))
        return self.root / path, float(row[f'videos/{key}/from_timestamp'])

    def image(self, episode_index, frame_index, name):
        path, offset = self.video(episode_index, name)
        target = offset + frame_index / 30
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is not None and float(frame.pts * stream.time_base) >= target - .5 / 30:
                    if abs(float(frame.pts * stream.time_base) - target) > 1 / 30:
                        raise ValueError(f'Video timestamp mismatch: {path}, {target}')
                    return frame.to_ndarray(format='rgb24')
        raise ValueError(f'No RGB frame at {path}:{target}')

    def sample(self, episode_index=0, frame_index=0, horizon=30):
        table = self.table(episode_index)
        if not 0 <= frame_index < len(table) or horizon < 1:
            raise ValueError('Invalid frame index/horizon')
        state = np.asarray(table.iloc[frame_index]['observation.state'], dtype=np.float32)
        actions = np.stack(table.action).astype(np.float32)
        # v4 action[i] already means q(i+1). Clamp only inside this episode.
        indices = np.minimum(np.arange(frame_index, frame_index + horizon), len(table) - 1)
        return state, tuple(self.image(episode_index, frame_index, c) for c in CAMERAS), actions[indices]

    def statistics(self, episodes=None):
        arrays = {'observation.state': [], 'action': []}
        for episode in (self.episodes.index if episodes is None else episodes):
            table = self.table(episode)
            for key in arrays:
                value = np.stack(table[key]).astype(np.float32)
                if value.shape != (len(table), 7) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid seven-joint data: episode={episode}, {key}')
                arrays[key].append(value)
        result = {}
        for key, values in arrays.items():
            array = np.concatenate(values)
            result[key] = dict(min=array.min(0).tolist(), max=array.max(0).tolist(),
                               mean=array.mean(0).tolist(), std=array.std(0).tolist(),
                               q01=np.quantile(array, .01, axis=0).tolist(),
                               q99=np.quantile(array, .99, axis=0).tolist(),
                               count=[len(array)])
        return result


def training_sample(dataset, model, episode_index, frame_index, horizon, stats):
    """Legacy full-export layouts. Current cloud training uses scripts/v4_data.py."""
    if model == 'wsa':
        from .wsa_data import training_sample as wsa_sample
        return wsa_sample(dataset, episode_index, frame_index, horizon)
    from .contract import normalize, pack_rdt, pad_xvla, hamlet_observation
    state, images, actions = dataset.sample(episode_index, frame_index, horizon)
    if model == 'xvla':
        return dict(proprio=pad_xvla(state), action=actions, domain_id=29,
                    images=images, language_instruction=dataset.task)
    if model == 'flower':
        return dict(proprio=np.pad(normalize(state, stats['observation.state']), (0, 1)),
                    actions=np.pad(normalize(actions, stats['action']), ((0, 0), (0, 1))),
                    rgb_obs={c: flower_image(image)[None] for c, image in zip(CAMERAS, images, strict=True)},
                    lang_text=dataset.task, action_type=0, frequency=30)
    if model == 'rdt':
        states, mask = pack_rdt(state)
        targets, _ = pack_rdt(actions)
        ordered = (images[1], images[0], images[2])
        # Both history slots use the current observation, identically at train and inference.
        return dict(state=states, actions=targets, action_mask=mask, ctrl_freq=30,
                    images=(*ordered, *ordered), language_instruction=dataset.task)
    if model == 'hamlet':
        return dict(observation=hamlet_observation(state, images, dataset.task),
                    actions={'joint_position': actions})
    if model == 'pi05':
        return {'observation.state': state, 'action': actions, 'task': dataset.task,
                **{f'observation.images.{c}': image for c, image in zip(CAMERAS, images, strict=True)}}
    raise ValueError(f'Unknown model: {model}')
