"""Weight-only initialization, grouped validation and deterministic DDP continuation."""
import copy
import hashlib
import json
import random
import time
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data.distributed import DistributedSampler


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_trainable(model, checkpoint):
    checkpoint = Path(checkpoint)
    assert (checkpoint / 'COMPLETE').is_file(), 'Incomplete initialization checkpoint'
    state = load_file(str(checkpoint / 'trainable.safetensors'))
    parameters = {k: v for k, v in model.named_parameters() if v.requires_grad}
    assert set(state) == set(parameters), 'Trainable parameter keys differ'
    with torch.no_grad():
        for name, parameter in parameters.items():
            assert parameter.shape == state[name].shape, name
            parameter.copy_(state[name])
    return dict(checkpoint=str(checkpoint), weights_sha256=file_sha256(checkpoint / 'trainable.safetensors'),
                loaded_parameter_tensors=len(state))


def parent_normalization(data, checkpoint):
    from v4_data import Normalizer
    config = json.loads((checkpoint / 'config.json').read_text())
    saved = json.loads((checkpoint / 'data_audit.json').read_text())
    assert config['model'] == data.model and config['normalization_mode'] == data.norm.mode
    original_hash = data.audit['data_sha256']
    stats = saved['normalization']
    for key in ['observation.state', 'action']:
        for name in ['mean', 'std', 'min', 'max', 'q01', 'q99']:
            assert np.asarray(stats[key][name]).shape == (7,) and np.isfinite(stats[key][name]).all()
        assert (np.asarray(stats[key]['max']) >= stats[key]['min']).all()
    data.audit['computed_training_normalization'] = copy.deepcopy(data.audit['normalization'])
    data.audit['normalization'] = copy.deepcopy(stats)
    data.audit['normalization_source'] = dict(checkpoint=str(checkpoint),
        audit_sha256=file_sha256(checkpoint / 'data_audit.json'),
        rationale='preserve pretrained state/action coordinate system; includes parent training data')
    data.audit['stats_scope'] = 'effective normalization inherited from initialization checkpoint'
    data.norm = Normalizer(data.audit['normalization'], config['normalization_mode'])
    assert data.audit['data_sha256'] == original_hash


class OffsetDistributedSampler(DistributedSampler):
    """Skip already consumed indices without decoding their videos on restart."""
    offset = 0

    def __iter__(self):
        return islice(super().__iter__(), self.offset, None)

    def __len__(self):
        return super().__len__() - self.offset

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.offset = 0


def capture_rng(device):
    return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                numpy_rng=np.random.get_state(), python_rng=random.getstate())


def restore_rng(state, device):
    torch.set_rng_state(state['torch_rng'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda_rng'], device)
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])


@contextmanager
def isolated_evaluation(model, device):
    states = [(module, module.training) for module in model.modules()]
    rng = capture_rng(device)
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in states:
            module.training = training
        restore_rng(rng, device)


def validation_anchors(data, held_out, count):
    if count >= len(held_out):
        return list(held_out)
    regions = data.audit['split'].get('episode_regions')
    if not regions:
        return np.asarray(held_out)[np.linspace(0, len(held_out)-1, count, dtype=int)].tolist()
    pools = {region: [] for region in ['left', 'center', 'right']}
    held_set = set(held_out)
    for ep in data.episodes.itertuples():
        region = regions.get(str(ep.episode_index))
        if region in pools:
            pools[region].extend(i for i in range(int(ep.dataset_from_index), int(ep.dataset_to_index)) if i in held_set)
    assert sum(map(len, pools.values())) == len(held_out)
    selected = []
    for index, pool in enumerate(pools.values()):
        quota = count // 3 + int(index < count % 3)
        assert len(pool) >= quota
        selected.extend(np.asarray(pool)[np.linspace(0, len(pool)-1, quota, dtype=int)].tolist())
    return sorted(selected)


def evaluate_hamlet(model, loader, data, device, seed, dist=None):
    """Fixed-seed flow loss and generated actions, aggregated over real joint coordinates."""
    started = time.time()
    # SSE, absolute error, valid coordinates, flow loss sum, samples; joint SSE/AE;
    # J1 |q| sum, J1 >15deg count; three region SSE/AE/count triples.
    totals = torch.zeros(30, dtype=torch.float64, device=device)
    regions = data.audit['split'].get('episode_regions', {})
    with isolated_evaluation(model, device):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            assert batch['action'].shape[0] == 1, 'HAMLET validation uses per-anchor RNG'
            anchor = int(batch['index'].item())
            sample_seed = seed + anchor
            torch.manual_seed(sample_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(sample_seed)
            model.model.action_head.reset_memory()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                flow = model(batch)
                torch.manual_seed(sample_seed + 1000000)
                if device.type == 'cuda':
                    torch.cuda.manual_seed(sample_seed + 1000000)
                normalized = model.predict(batch)
            prediction = data.norm.undo(normalized.float())
            target = data.norm.undo(batch['action'].float())
            assert prediction.shape == target.shape and torch.isfinite(prediction).all() and torch.isfinite(flow)
            valid = (~batch['action_is_pad'])[..., None].expand_as(target)
            delta = prediction.double() - target.double()
            squared = torch.where(valid, delta.square(), 0)
            absolute = torch.where(valid, delta.abs(), 0)
            count = valid.sum()
            totals[0] += squared.sum(); totals[1] += absolute.sum(); totals[2] += count
            totals[3] += flow.double() * count; totals[4] += 1
            totals[5:12] += squared.sum((0, 1)); totals[12:19] += absolute.sum((0, 1))
            q1 = prediction[..., 0][valid[..., 0]]
            totals[19] += q1.double().abs().sum()
            totals[20] += (q1.abs() > np.deg2rad(15)).sum()
            region = regions.get(str(int(batch['episode_index'].item())))
            if region in ['left', 'center', 'right']:
                i = 21 + 3 * ['left', 'center', 'right'].index(region)
                totals[i:i+3] += torch.stack([squared.sum(), absolute.sum(), count.double()])
        model.model.action_head.reset_memory()
        if dist is not None:
            dist.all_reduce(totals)
    values = totals.cpu().tolist()
    assert values[2] > 0
    joint_count = values[2] / 7
    result = dict(val_action_mse_rad2=values[0]/values[2], val_action_mae_rad=values[1]/values[2],
        val_loss=values[3]/values[2], validation_samples=int(values[4]), validation_valid_coordinates=int(values[2]),
        per_joint_mse_rad2=[x/joint_count for x in values[5:12]], per_joint_mae_rad=[x/joint_count for x in values[12:19]],
        j1_predicted_abs_mean_rad=values[19]/joint_count, j1_predicted_outside15_fraction=values[20]/joint_count,
        validation_seconds=time.time()-started)
    result['regions'] = {name: dict(mse_rad2=values[i]/values[i+2], mae_rad=values[i+1]/values[i+2],
                                  valid_coordinates=int(values[i+2]))
                         for name, i in zip(['left', 'center', 'right'], [21, 24, 27]) if values[i+2]}
    return result


def record_best(output, checkpoint, step, metrics, baseline=False):
    key = 'val_action_mse_rad2'
    candidate = dict(checkpoint=str(checkpoint), step=step, metric=key, loss=metrics[key],
                     is_initial_baseline=baseline, validation=metrics,
                     limitation='Held out from this fine-tuning run; parent model previously trained on this old data')
    for name in (['best.json'] if baseline else ['best.json', 'best_new.json']):
        path = output/name
        if not path.exists() or candidate['loss'] < json.loads(path.read_text())['loss']:
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(candidate, indent=2)+'\n')
            temporary.replace(path)
