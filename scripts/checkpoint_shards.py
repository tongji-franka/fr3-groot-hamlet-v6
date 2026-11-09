"""Complete ZeRO-1 checkpoints with rank-local AdamW states and atomic publication."""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from training_support import capture_rng, restore_rng, load_trainable

FORMAT = 'zero1_local_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sync_options(optimizer):
    assert len(optimizer.param_groups) == len(optimizer.optim.param_groups)
    for outer, local in zip(optimizer.param_groups, optimizer.optim.param_groups):
        for key, value in outer.items():
            if key != 'params':
                local[key] = value


def parameter_layout(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.optim.param_groups]


def durable(path):
    with Path(path).open('rb') as f:
        os.fsync(f.fileno())


def durable_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_sharded(model, optimizer, scheduler, step, destination, config, audit, training_root, device, dist):
    from fr3_vla.cloud import snapshot_training_runtime
    destination = Path(destination)
    staging = destination.with_name(destination.name + '.partial')
    rank, world = dist.get_rank(), dist.get_world_size()
    started = time.perf_counter()
    if rank == 0:
        assert not destination.exists() and not staging.exists(), 'Refusing checkpoint overwrite'
        staging.mkdir(parents=True)
    dist.barrier()
    sync_options(optimizer)
    shard_name = f'optimizer-rank-{rank:03d}.pt'
    shard = dict(format=FORMAT, step=step, rank=rank, world_size=world,
        parameter_layout=parameter_layout(model, optimizer), optimizer=optimizer.optim.state_dict(),
        global_group_options=[{k: copy.deepcopy(v) for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
        scheduler=scheduler.state_dict(), rng=capture_rng(device))
    shard_started = time.perf_counter()
    torch.save(shard, staging/shard_name)
    durable(staging/shard_name)
    shard_hash = sha(staging/shard_name)
    (staging/(shard_name+'.sha256')).write_text(shard_hash+'\n')
    local_seconds = time.perf_counter()-shard_started
    model_seconds = 0.
    if rank == 0:
        model_started = time.perf_counter()
        params = {k: v.detach().cpu().contiguous() for k,v in model.named_parameters() if v.requires_grad}
        save_file(params, str(staging/'trainable.safetensors'))
        durable(staging/'trainable.safetensors')
        (staging/'trainable.safetensors.sha256').write_text(sha(staging/'trainable.safetensors')+'\n')
        del params
        saved_config = {**config, 'checkpoint_optimizer_format': FORMAT}
        (staging/'config.json').write_text(json.dumps(saved_config, indent=2))
        (staging/'data_audit.json').write_text(json.dumps(audit, indent=2))
        if audit.get('split'):
            (staging/'split.json').write_text(json.dumps(audit['split'], indent=2))
        snapshot_training_runtime(staging, saved_config, audit, training_root)
        model_seconds = time.perf_counter()-model_started
    timings = torch.tensor([local_seconds], dtype=torch.float64, device=device)
    dist.all_reduce(timings, op=dist.ReduceOp.MAX)
    dist.barrier()
    if rank == 0:
        names = [f'optimizer-rank-{i:03d}.pt' for i in range(world)]
        assert all((staging/name).is_file() and (staging/(name+'.sha256')).is_file() for name in names)
        torch.save(dict(format=FORMAT, step=step, world_size=world, shards=names), staging/'trainer.pt')
        durable(staging/'trainer.pt')
        for path in staging.rglob('*'):
            if path.is_file():
                durable(path)
        for path in staging.rglob('*'):
            if path.is_dir():
                durable_directory(path)
        (staging/'COMPLETE').touch()
        durable(staging/'COMPLETE')
        durable_directory(staging)
        staging.rename(destination)
        durable_directory(destination.parent)
    dist.barrier()
    return dict(checkpoint=str(destination), checkpoint_seconds=time.perf_counter()-started,
                max_rank_shard_write_hash_seconds=timings.item(), rank0_model_metadata_seconds=model_seconds,
                format=FORMAT, rank=rank, world_size=world)


def load_sharded(model, optimizer, scheduler, checkpoint, device, dist, *, load_weights=True):
    checkpoint = Path(checkpoint)
    assert (checkpoint/'COMPLETE').is_file()
    metadata = torch.load(checkpoint/'trainer.pt', map_location='cpu', weights_only=False)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert metadata['format']==FORMAT and metadata['world_size']==world, 'Optimizer shards require original world size'
    path = checkpoint/f'optimizer-rank-{rank:03d}.pt'
    assert sha(path)==path.with_suffix(path.suffix+'.sha256').read_text().strip(), 'Optimizer shard checksum mismatch'
    if load_weights:
        assert sha(checkpoint/'trainable.safetensors')==(checkpoint/'trainable.safetensors.sha256').read_text().strip()
        load_trainable(model, checkpoint)
    shard = torch.load(path, map_location='cpu', weights_only=False)
    assert shard['format']==FORMAT and shard['rank']==rank and shard['world_size']==world
    assert shard['step']==metadata['step']
    assert shard['parameter_layout']==parameter_layout(model, optimizer), 'Local optimizer parameter partition changed'
    optimizer.optim.load_state_dict(shard['optimizer'])
    if optimizer.state is not optimizer.optim.state:
        optimizer.state.clear()
    for group, options in zip(optimizer.param_groups, shard['global_group_options']):
        group.update(options)
    sync_options(optimizer)
    scheduler.load_state_dict(shard['scheduler'])
    restore_rng(shard['rng'], device)
    return metadata['step']
