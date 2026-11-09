"""FR3 joint-position data contract shared by the five fine-tuning adapters."""
import hashlib
import json
from bisect import bisect_right
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/root/franka-hockey/data/hockey_three_cameras_aligned_v4')
CAMERAS = ['observation.images.overhead', 'observation.images.wrist', 'observation.images.third']
JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]
HORIZONS = {'xvla': 30, 'flower': 20, 'hamlet': 50, 'rdt': 64, 'pi05': 50}
MODES = {'xvla': 'mean_std', 'flower': 'bounds', 'hamlet': 'bounds', 'rdt': 'mean_std', 'pi05': 'quantiles'}


def masked_mean(errors, valid):
    valid = valid.expand_as(errors)
    if not valid.any():
        raise ValueError('Batch has no supervised action coordinates')
    return torch.where(valid, errors, 0).sum() / valid.sum()


def pad_last(x, dim):
    if x.shape[-1] > dim:
        raise ValueError(f'Cannot fit {x.shape[-1]} coordinates into {dim}')
    return F.pad(x, (0, dim - x.shape[-1]))


def resize_pad(image, size=224):
    h, w = image.shape[-2:]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    single = image.ndim == 3
    image = F.interpolate(image.unsqueeze(0) if single else image, (nh, nw), mode='bilinear',
                          align_corners=False, antialias=True)
    # Antialiasing can round a white pixel slightly above one.
    image.clamp_(0, 1)
    if single:
        image = image.squeeze(0)
    return F.pad(image, ((size - nw) // 2, (size - nw + 1) // 2,
                         (size - nh) // 2, (size - nh + 1) // 2))


class Normalizer:
    def __init__(self, stats, mode):
        self.stats, self.mode = stats, mode

    def parameters(self, key, x):
        s = self.stats[key]
        if self.mode == 'mean_std':
            center, scale = s['mean'], np.maximum(s['std'], 1e-3)
        else:
            lo, hi = ('q01', 'q99') if self.mode == 'quantiles' else ('min', 'max')
            center = (np.asarray(s[lo]) + np.asarray(s[hi])) / 2
            scale = np.maximum((np.asarray(s[hi]) - np.asarray(s[lo])) / 2, 1e-3)
        return x.new_tensor(center), x.new_tensor(scale)

    def apply(self, x, key):
        center, scale = self.parameters(key, x)
        return (x - center) / scale

    def undo(self, x, key='action'):
        center, scale = self.parameters(key, x)
        return x * scale + center


def dataset_roots(root):
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    roots = [Path(p).expanduser().resolve(strict=True) for p in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('Specify one or more distinct dataset roots')
    return roots


def episode_indices(episodes, selected):
    selected = set(selected)
    return [i for ep in episodes.itertuples() if int(ep.episode_index) in selected
            for i in range(int(ep.dataset_from_index), int(ep.dataset_to_index))]


def split_episodes(sources, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    rng = np.random.default_rng(seed)
    train, validation, counts = [], [], []
    train_groups, validation_groups, seen = [], [], set()
    offset = 0
    for source in sources:
        count = source['episodes']
        if count < 2:
            raise ValueError('Each source needs at least two episodes')
        ids = np.arange(offset, offset + count)
        groups = source.get('episode_groups', [f'episode:{i}' for i in ids])
        unique = sorted(set(groups))
        if len(groups) != count or len(unique) < 2 or seen & set(unique):
            raise ValueError('Invalid recording groups, or groups shared between dataset sources')
        seen.update(unique)
        held_groups = set(rng.choice(unique, min(len(unique) - 1, max(1, round(len(unique) * fraction))), replace=False).tolist())
        held_out = [int(i) for i, group in zip(ids, groups) if group in held_groups]
        training = sorted(set(ids.tolist()) - set(held_out))
        train_groups.extend(sorted(set(unique) - held_groups))
        validation_groups.extend(sorted(held_groups))
        train.extend(training)
        validation.extend(held_out)
        counts.append(dict(dataset=source['dataset'], train_episodes=len(training), validation_episodes=len(held_out),
                           train_groups=len(unique) - len(held_groups), validation_groups=len(held_groups)))
        offset += count
    assert not set(train) & set(validation) and set(train) | set(validation) == set(range(offset))
    return dict(version=1, seed=seed, validation_fraction=fraction,
                grouping='raw recording groups, stratified by source; derived episodes stay together',
                train_episodes=train, validation_episodes=validation,
                train_groups=train_groups, validation_groups=validation_groups, sources=counts)


def explicit_split(sources, manifest):
    """Honor reviewed episode IDs while keeping every original recording together."""
    requested = json.loads(Path(manifest).read_text())
    assert requested['dataset_names'] == [Path(s['dataset']).name for s in sources]
    assert requested['source_frames'] == [s['frames'] for s in sources]
    total = sum(s['episodes'] for s in sources)
    train = [int(i) for i in requested['train_episodes']]
    validation = [int(i) for i in requested['validation_episodes']]
    assert train and validation
    assert len(set(train)) == len(train) and len(set(validation)) == len(validation)
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(total))
    groups = [g for source in sources for g in source['episode_groups']]
    train_groups, validation_groups = {groups[i] for i in train}, {groups[i] for i in validation}
    assert not train_groups & validation_groups, 'Raw recording leakage across splits'
    counts, offset = [], 0
    for source in sources:
        ids = set(range(offset, offset + source['episodes']))
        tr, va = ids & set(train), ids & set(validation)
        counts.append(dict(dataset=source['dataset'], train_episodes=len(tr), validation_episodes=len(va),
                           train_groups=len({groups[i] for i in tr}), validation_groups=len({groups[i] for i in va})))
        offset += source['episodes']
    return dict(version=2, seed=requested['seed'], grouping='explicit reviewed original-recording groups',
                train_episodes=sorted(train), validation_episodes=sorted(validation),
                train_groups=sorted(train_groups), validation_groups=sorted(validation_groups), sources=counts,
                episode_regions=requested.get('episode_regions', {}),
                manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                limitation=requested.get('limitation', ''))


def audit(root=DATA, *, output_path=ROOT / 'reports/data_audit.json', validation_fraction=0., split_seed=42,
          split_manifest=None):
    assert not (validation_fraction and split_manifest), 'Choose a fraction or an explicit split'
    roots = dataset_roots(root)
    sources, all_frames, all_episodes = [], [], []
    digest = hashlib.sha256()
    episode_count = 0
    for root in roots:
        info = json.loads((root / 'meta/info.json').read_text())
        assert info['fps'] == 30 and info['robot_type'] == 'fr3_teach_aligned_single_arm'
        for key in ['observation.state', 'action']:
            assert info['features'][key]['names'] == JOINTS
            assert info['features'][key]['shape'] == [7]
        for key in CAMERAS:
            assert info['features'][key]['dtype'] == 'video'
        episodes = pd.concat([pd.read_parquet(p) for p in sorted((root / 'meta/episodes').rglob('*.parquet'))])
        frames = pd.concat([pd.read_parquet(p) for p in sorted((root / 'data').rglob('*.parquet'))], ignore_index=True)
        assert len(frames) == info['total_frames'] and len(episodes) == info['total_episodes']
        assert np.array_equal(frames['index'], np.arange(len(frames)))
        assert np.array_equal(episodes['episode_index'], np.arange(len(episodes)))
        assert int(episodes.iloc[0].dataset_from_index) == 0
        assert int(episodes.iloc[-1].dataset_to_index) == len(frames)
        for ep in episodes.itertuples():
            start, end = int(ep.dataset_from_index), int(ep.dataset_to_index)
            part = frames.iloc[start:end]
            assert len(part) == ep.length and (part['episode_index'] == ep.episode_index).all()
            assert np.array_equal(part['frame_index'], np.arange(ep.length))
            assert np.allclose(part['timestamp'], np.arange(ep.length) / info['fps'], atol=1e-4)
        source_digest = hashlib.sha256()
        for path in sorted((root / 'data').rglob('*.parquet')):
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
                    source_digest.update(block)
        sources.append(dict(dataset=str(root), episodes=len(episodes), frames=len(frames),
                            data_sha256=source_digest.hexdigest()))
        if validation_fraction or split_manifest:
            provenance = json.loads((root / 'cleaning/provenance.json').read_text())
            mapping = {int(item['episode_index']): item['source_group_id'] for item in provenance['episode_mapping']}
            sources[-1]['episode_groups'] = [mapping[i] for i in range(len(episodes))]
        ep = episodes.copy()
        ep['episode_index'] += episode_count
        frame_offset = sum(len(part) for part in all_frames)
        ep['dataset_from_index'] += frame_offset
        ep['dataset_to_index'] += frame_offset
        all_episodes.append(ep)
        episode_count += len(episodes)
        all_frames.append(frames)
    frames = pd.concat(all_frames, ignore_index=True)
    split = None
    selected = np.arange(len(frames))
    if validation_fraction or split_manifest:
        split = explicit_split(sources, split_manifest) if split_manifest else split_episodes(sources, validation_fraction, split_seed)
        episodes = pd.concat(all_episodes, ignore_index=True)
        selected = np.asarray(episode_indices(episodes, split['train_episodes']), dtype=np.int64)
        split['train_frames'] = len(selected)
        split['validation_frames'] = len(frames) - len(selected)
        split['data_sha256'] = digest.hexdigest()
    stats = {}
    arrays = {}
    for key in ['observation.state', 'action']:
        x = np.stack(frames[key]).astype(np.float64)
        assert x.shape == (len(frames), 7) and np.isfinite(x).all()
        x = x[selected]
        arrays[key] = x
        stats[key] = {k: v.tolist() for k, v in {
            'mean': x.mean(0), 'std': x.std(0), 'min': x.min(0), 'max': x.max(0),
            'q01': np.quantile(x, .01, axis=0), 'q99': np.quantile(x, .99, axis=0),
        }.items()}
    for mode in set(MODES.values()):
        norm = Normalizer(stats, mode)
        for key, x in arrays.items():
            t = torch.from_numpy(x.astype(np.float32))
            assert torch.allclose(norm.undo(norm.apply(t, key), key), t, atol=1e-6)
    errors = torch.tensor([[[1., 3., 99.], [99., 99., 99.]]], requires_grad=True)
    valid = torch.tensor([[[True, True, False], [False, False, False]]])
    loss = masked_mean(errors, valid)
    loss.backward()
    assert loss.item() == 2 and torch.equal(errors.grad[~valid], torch.zeros(4))
    report = {'dataset': str(roots[0]) if len(roots) == 1 else [str(p) for p in roots],
              'datasets': sources, 'episodes': episode_count, 'frames': len(frames), 'fps': info['fps'],
              'joint_names': JOINTS, 'action_semantics': 'absolute joint positions, radians, no gripper',
              'cameras': CAMERAS, 'normalization': stats, 'normalization_modes': MODES,
              'data_sha256': digest.hexdigest(), 'stats_scope': f'all {episode_count} training episodes',
              'action_state_abs_error_mean': np.abs(arrays['action'] - arrays['observation.state']).mean(0).tolist(),
              'mask_gradient_test': 'passed', 'normalization_roundtrip': 'passed'}
    if split is not None:
        report['split'] = split
        report['split_sha256'] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
        report['stats_scope'] = f"training split only: {len(split['train_episodes'])} episodes, {len(selected)} frames"
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
    return report


class V4Dataset(Dataset):
    def __init__(self, model, root=DATA, *, audit_report=None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.model, self.horizon = model, HORIZONS[model]
        self.image_size = 384 if model == 'rdt' else 224
        self.history = [-48, -32, -16, 0] if model == 'hamlet' else ([-1, 0] if model == 'rdt' else [0])
        deltas = {'action': [i / 30 for i in range(self.horizon)]}
        deltas.update({k: [i / 30 for i in self.history] for k in CAMERAS})
        roots = dataset_roots(root)
        datasets, episodes = [], []
        self.episode_offsets = []
        frame_offset, episode_offset = 0, 0
        for root in roots:
            ds = LeRobotDataset(f'tongji-franka/{root.name}', root=root,
                                delta_timestamps=deltas, video_backend='pyav',
                                image_transforms=partial(resize_pad, size=self.image_size))
            ep = pd.concat([pd.read_parquet(p) for p in sorted((root / 'meta/episodes').rglob('*.parquet'))])
            self.episode_offsets.append(episode_offset)
            ep['episode_index'] += episode_offset
            ep['dataset_from_index'] += frame_offset
            ep['dataset_to_index'] += frame_offset
            episode_offset += len(ep)
            frame_offset += len(ds)
            datasets.append(ds)
            episodes.append(ep)
        self.ds = ConcatDataset(datasets)
        self.episodes = pd.concat(episodes, ignore_index=True)
        if audit_report is None:
            audit_report = (json.loads((ROOT / 'reports/data_audit.json').read_text())
                            if roots == [DATA] else audit(roots, output_path=None))
        self.audit = audit_report
        assert self.audit['frames'] == len(self.ds) and self.audit['episodes'] == len(self.episodes)
        assert self.audit['dataset'] == (str(roots[0]) if len(roots) == 1 else [str(p) for p in roots])
        self.norm = Normalizer(self.audit['normalization'], MODES[model])

    def __len__(self):
        return len(self.ds)

    def restore_stats(self, checkpoint, *, allow_source_subset=False):
        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / 'config.json').read_text())
        saved = json.loads((checkpoint / 'data_audit.json').read_text())
        assert config['model'] == self.model
        hashes = {saved['data_sha256']}
        if allow_source_subset:
            hashes.update(source['data_sha256'] for source in saved.get('datasets', []))
        assert self.audit['data_sha256'] in hashes
        self.audit = saved
        self.norm = Normalizer(saved['normalization'], config['normalization_mode'])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        dataset_index = bisect_right(self.ds.cumulative_sizes, index)
        item = self.ds[index]
        action, state = item['action'].float(), item['observation.state'].float()
        action_pad = item['action_is_pad'].bool()
        assert action.shape == (self.horizon, 7) and state.shape == (7,)
        assert not action_pad[0] and torch.isfinite(action).all() and torch.isfinite(state).all()
        images = torch.stack([item[k].unsqueeze(0) if item[k].ndim == 3 else item[k] for k in CAMERAS], 1)
        assert images.shape == (len(self.history), 3, 3, self.image_size, self.image_size)
        return {'action': self.norm.apply(action, 'action'),
                'state': self.norm.apply(state, 'observation.state'), 'images': images,
                'action_is_pad': action_pad, 'task': item['task'],
                'index': index, 'episode_index': int(item['episode_index']) + self.episode_offsets[dataset_index]}

    def boundary_check(self):
        rows = []
        for ep_id in [0, len(self.episodes) // 2, len(self.episodes) - 1]:
            ep = self.episodes.iloc[ep_id]
            for idx in [int(ep.dataset_from_index), int(ep.dataset_to_index) - 1]:
                item = self[idx]
                count = min(self.horizon, int(ep.dataset_to_index) - idx)
                assert (~item['action_is_pad']).sum().item() == count
                assert item['episode_index'] == int(ep.episode_index)
                if count < self.horizon:
                    assert torch.equal(item['action'][count:], item['action'][count - 1:count].expand(self.horizon - count, -1))
                assert item['images'].min() >= 0 and item['images'].max() <= 1
                rows.append({'index': idx, 'episode': int(ep.episode_index), 'valid_actions': count})
        return rows


if __name__ == '__main__':
    result = audit()
    print(json.dumps({k: v for k, v in result.items() if k != 'normalization'}, indent=2))
