"""Fine-tune the official backbones on the LeRobot v4 dataset."""
import argparse
import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from safetensors.torch import load_file, save_file

from v4_data import ROOT, DATA, V4Dataset, HORIZONS, MODES, CAMERAS, audit, episode_indices
from training_support import (OffsetDistributedSampler, capture_rng, restore_rng, load_trainable,
                              parent_normalization, validation_anchors, evaluate_hamlet, record_best)

sys.path.insert(0, str(ROOT))
from fr3_vla.cloud import snapshot_training_runtime


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def evaluate_loss(model, loader, device, seed, dist=None):
    """Evaluate held-out frames without advancing training RNG or changing module modes."""
    modes = [(module, module.training) for module in model.modules()]
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    started = time.time()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    try:
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model.eval()
            for batch in loader:
                batch = move(batch, device)
                coordinates = int((~batch['action_is_pad']).sum()) * batch['action'].shape[-1]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    loss = model(batch)
                assert torch.isfinite(loss) and coordinates > 0
                totals[0] += loss.double() * coordinates
                totals[1] += coordinates
                totals[2] += len(batch['action'])
            if dist is not None:
                dist.all_reduce(totals)
        assert totals[1] > 0
        return dict(val_loss=(totals[0] / totals[1]).item(),
                    validation_samples=int(totals[2]), validation_valid_coordinates=int(totals[1]),
                    validation_seconds=time.time() - started)
    finally:
        for module, training in modes:
            module.training = training
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def save_checkpoint(model, optimizer, scheduler, step, output, config, data_audit, rank_rng_states=None):
    dest = output / f'checkpoint-{step:06d}'
    dest.mkdir(parents=True, exist_ok=False)
    params = {k: v.detach().cpu().contiguous() for k, v in model.named_parameters() if v.requires_grad}
    save_file(params, str(dest / 'trainable.safetensors'))
    torch.save({'step': step, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                'rank_rng_states': rank_rng_states}, dest / 'trainer.pt')
    (dest / 'config.json').write_text(json.dumps(config, indent=2))
    (dest / 'data_audit.json').write_text(json.dumps(data_audit, indent=2))
    if data_audit.get('split'):
        (dest / 'split.json').write_text(json.dumps(data_audit['split'], indent=2))
    snapshot_training_runtime(dest, config, data_audit, ROOT)
    (dest / 'COMPLETE').touch()
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(HORIZONS))
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--init-from', type=Path, help='weights and normalization only; new optimizer and step zero')
    p.add_argument('--split-manifest', type=Path, help='reviewed group-safe train/validation episode IDs')
    p.add_argument('--warmup-steps', type=int)
    p.add_argument('--stop-after', type=int, help='save and stop at this step, retaining the full --steps schedule')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--rtc-max-delay', default='auto',
                   help="'auto' resolves to 10 for pi05 and 0 otherwise; an int in [0, chunk_size); 0 disables")
    p.add_argument('--lora', action='store_true', help='pi05 only: LoRA on the gemma_expert instead of full fine-tuning')
    p.add_argument('--lora-r', type=int, default=32)
    p.add_argument('--lora-alpha', type=int, default=64)
    p.add_argument('--ddp', action='store_true', help='multi-GPU data parallel via torchrun')
    p.add_argument('--zero-optimizer', action='store_true', help='shard AdamW states across DDP ranks')
    p.add_argument('--sharded-checkpoint', action='store_true', help='save rank-local ZeRO optimizer shards in parallel')
    p.add_argument('--dataset-root', type=Path, nargs='+', help='one or more local LeRobot dataset roots')
    p.add_argument('--validation-fraction', type=float, default=0.)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--eval-samples', type=int, default=0,
                   help='fixed evenly spaced held-out frames; 0 evaluates every held-out frame')
    p.add_argument('--report-suffix', default='', help='suffix for the smoke report filename (e.g. _rtc10)')
    args = p.parse_args()
    assert not (args.ddp and args.smoke), 'The smoke self-check path is single-GPU only'
    if args.zero_optimizer and not args.ddp:
        p.error('--zero-optimizer requires --ddp')
    if args.sharded_checkpoint and (not args.zero_optimizer or args.smoke):
        p.error('--sharded-checkpoint requires distributed ZeRO and non-smoke training')
    if min(args.steps, args.batch_size, args.grad_accum, args.save_every) < 1:
        p.error('steps, batch-size, grad-accum and save-every must be positive')
    if not 0 <= args.validation_fraction < 1 or args.eval_every < 1 or args.eval_samples < 0:
        p.error('Invalid validation fraction, evaluation interval, or sample count')
    if args.resume and args.init_from:
        p.error('--resume and --init-from are mutually exclusive')
    if args.validation_fraction and args.split_manifest:
        p.error('Choose --validation-fraction or --split-manifest')
    has_validation = bool(args.validation_fraction or args.split_manifest)
    if has_validation and (args.smoke or args.model not in {'xvla', 'flower', 'hamlet'}):
        p.error('Validation supports non-smoke X-VLA, FLOWER and HAMLET runs')
    if has_validation and args.model == 'hamlet' and args.batch_size != 1:
        p.error('HAMLET validation requires batch-size=1 for per-anchor seeds')
    if args.warmup_steps is not None and not 0 <= args.warmup_steps < args.steps:
        p.error('warmup-steps must be in [0, steps)')
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        p.error('stop-after must be in (0, steps]')

    rtc = args.rtc_max_delay
    if rtc == 'auto':
        rtc = 10 if args.model == 'pi05' else 0
    rtc = int(rtc)

    rank, world, local = 0, 1, 0
    dist = None
    if args.ddp:
        import torch.distributed as dist
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local)
    torch.set_num_threads(4)
    seed = 42 + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    device = torch.device('cuda', local) if args.ddp else torch.device('cuda')
    if rank == 0:
        if args.output.exists() and not args.resume:
            raise FileExistsError(f'Refusing to overwrite {args.output}; use a new output or --resume')
        args.output.mkdir(parents=True, exist_ok=args.resume is not None)
    if dist is not None:
        dist.barrier()
    if args.dataset_root or has_validation:
        audit_path = args.output / 'data_audit.json'
        if rank == 0:
            audit(args.dataset_root or DATA, output_path=audit_path,
                  validation_fraction=args.validation_fraction, split_seed=args.split_seed,
                  split_manifest=args.split_manifest)
        if dist is not None:
            dist.barrier()
        data = V4Dataset(args.model, root=args.dataset_root or DATA,
                         audit_report=json.loads(audit_path.read_text()))
    else:
        data = V4Dataset(args.model)
    if args.resume:
        saved_audit = json.loads((args.resume / 'data_audit.json').read_text())
        assert saved_audit.get('split_sha256') == data.audit.get('split_sha256'), 'Requested split differs from checkpoint'
        data.restore_stats(args.resume)
    if args.init_from:
        assert (args.init_from / 'COMPLETE').is_file()
        parent_normalization(data, args.init_from)
    if rank == 0:
        (args.output / 'data_audit.json').write_text(json.dumps(data.audit, indent=2))
    boundary = data.boundary_check() if args.smoke else None
    # Force real episode ends into the training smoke test, where most targets are masked.
    smoke_episodes = [0, len(data.episodes) // 2, len(data.episodes) - 1]
    smoke_indices = [int(data.episodes.iloc[i].dataset_to_index) - 1 for i in smoke_episodes]
    smoke_indices += [int(data.episodes.iloc[i].dataset_from_index) for i in smoke_episodes]
    train_data = Subset(data, smoke_indices) if args.smoke else data
    validation_loader = None
    split = data.audit.get('split')
    if split:
        training_indices = episode_indices(data.episodes, split['train_episodes'])
        held_out = episode_indices(data.episodes, split['validation_episodes'])
        assert not set(training_indices) & set(held_out)
        assert len(training_indices) + len(held_out) == len(data)
        train_data = Subset(data, training_indices)
        count = min(len(held_out), args.eval_samples or len(held_out))
        validation_indices = validation_anchors(data, held_out, count)
        # Striding avoids DistributedSampler padding/duplicating validation examples.
        validation_loader = DataLoader(Subset(data, validation_indices[rank::world]),
                                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                       persistent_workers=args.num_workers > 0,
                                       multiprocessing_context='spawn' if args.num_workers > 0 else None,
                                       generator=torch.Generator().manual_seed(args.split_seed + rank))
        if rank == 0:
            (args.output / 'split.json').write_text(json.dumps(split, indent=2))
            (args.output / 'validation_indices.json').write_text(json.dumps(validation_indices))
    sampler = None
    if args.ddp:
        sampler = OffsetDistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=not args.smoke,
                                     seed=42, drop_last=True)
        loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True,
                            persistent_workers=args.num_workers > 0,
                            multiprocessing_context='spawn' if args.num_workers > 0 else None,
                            generator=torch.Generator().manual_seed(seed))
    else:
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=not args.smoke,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True,
                            persistent_workers=args.num_workers > 0,
                            multiprocessing_context='spawn' if args.num_workers > 0 else None,
                            generator=torch.Generator().manual_seed(seed))
    from adapters import ADAPTERS
    if args.model == 'pi05':
        model = ADAPTERS[args.model](lora=args.lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                     rtc_max_delay=rtc)
    else:
        assert not args.lora, '--lora is implemented for pi05 only'
        assert rtc == 0, f'{args.model} does not support training-time RTC'
        model = ADAPTERS[args.model]()
    initialization = None
    if args.init_from:
        old = json.loads((args.init_from / 'config.json').read_text())
        assert old['model'] == args.model
        assert bool(old.get('lora', False)) == args.lora and int(old.get('rtc_max_delay', 0)) == rtc
        initialization = load_trainable(model, args.init_from)
    model = model.to(device)
    raw = model
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            raw, device_ids=[local], broadcast_buffers=False, find_unused_parameters=args.lora or args.model == 'flower',
            gradient_as_bucket_view=args.zero_optimizer)
    model.train()
    lr = args.lr or {'xvla': 1e-4, 'flower': 2e-5, 'hamlet': 1e-4, 'rdt': 1e-4, 'pi05': 2.5e-5}[args.model]
    trainable = [p for p in raw.parameters() if p.requires_grad]
    assert trainable
    optimizer_options = dict(lr=lr, betas=(.9, .95), weight_decay=.01, foreach=False)
    if args.zero_optimizer:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(trainable, optimizer_class=torch.optim.AdamW, **optimizer_options)
    else:
        optimizer = torch.optim.AdamW(trainable, **optimizer_options)
    warmup = 0 if args.smoke else (args.warmup_steps if args.warmup_steps is not None else min(1000, args.steps // 10))
    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, args.steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_step = 0
    config = {**vars(args), 'rtc_max_delay': rtc, 'rtc_training': rtc > 0, 'lr': lr,
              'split_sha256': data.audit.get('split_sha256'), 'initialization': initialization,
              'effective_batch': args.batch_size * args.grad_accum * world, 'warmup_steps_effective': warmup,
              'validation_loss_aggregation': 'sum of squared errors / valid action coordinates',
              'data_path': data.audit['dataset'], 'data_sha256': data.audit['data_sha256'],
              'normalization_mode': data.norm.mode, 'cameras': CAMERAS, 'action_horizon': HORIZONS[args.model],
              'action_space': '7 absolute FR3 joint positions in radians; no gripper',
              'load_report': raw.load_report, 'trainable_parameters': sum(p.numel() for p in trainable),
              'total_parameters': sum(p.numel() for p in raw.parameters()),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device),
              'world_size': world, 'seed': seed,
              'ddp_resume_note': 'Per-rank RNG and exact consumed sampler indices restored for new checkpoints' if args.ddp else '',
              'weights': json.loads((ROOT / 'reports' / f'download_{args.model}.json').read_text()),
              'source_sha256': {str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest()
                                for s in (ROOT / 'scripts').glob('*.py')}}
    config = json.loads(json.dumps(config, default=str))
    if args.resume:
        assert (args.resume / 'COMPLETE').exists(), 'Incomplete checkpoint'
        old = json.loads((args.resume / 'config.json').read_text())
        assert old['model'] == args.model and old['data_sha256'] == data.audit['data_sha256']
        assert old.get('split_sha256') == config['split_sha256'], 'Training/validation split differs from checkpoint'
        assert bool(old.get('lora', False)) == args.lora and int(old.get('rtc_max_delay', 0)) == rtc
        assert bool(old.get('zero_optimizer', False)) == args.zero_optimizer, 'Optimizer mode differs from checkpoint'
        if args.ddp:
            assert old.get('world_size', 1) == world, 'DDP resume requires the same world size'
        if old.get('initialization'):
            config['initialization'] = old['initialization']
        if old.get('checkpoint_optimizer_format') == 'zero1_local_v1':
            from checkpoint_shards import load_sharded
            start_step = load_sharded(raw, optimizer, scheduler, args.resume, device, dist)
        else:
            load_trainable(raw, args.resume)
            training = torch.load(args.resume / 'trainer.pt', map_location='cpu', weights_only=False)
            optimizer.load_state_dict(training['optimizer'])
            if args.zero_optimizer:
                # PyTorch 2.8 also copies local moments into the unused outer state.
                optimizer.state.clear()
            scheduler.load_state_dict(training['scheduler'])
            start_step = training['step']
            if training.get('rank_rng_states'):
                assert len(training['rank_rng_states']) == world
                restore_rng(training['rank_rng_states'][rank], device)
            elif not args.ddp:
                torch.set_rng_state(training['torch_rng'])
                torch.cuda.set_rng_state_all(training['cuda_rng'])
                np.random.set_state(training['numpy_rng'])
                random.setstate(training['python_rng'])
            del training
    if rank == 0:
        (args.output / 'run_config.json').write_text(json.dumps(config, indent=2))
        print(json.dumps(config), flush=True)
    # Continue the sampler's permutation from where the interrupted run left off, so a resumed
    # run does not replay epoch-0 data order for its remaining steps.
    epoch = 0
    if args.resume and sampler is not None:
        full_epoch_batches = len(loader)
        epoch, consumed_batches = divmod(start_step * args.grad_accum, full_epoch_batches)
        sampler.set_epoch(epoch)
        sampler.offset = consumed_batches * args.batch_size
    run_end = min(args.steps, args.stop_after or args.steps)
    assert start_step < run_end, 'No additional optimizer steps requested'
    def validate():
        if args.model == 'hamlet':
            return evaluate_hamlet(raw, validation_loader, data, device, args.split_seed + 10000, dist)
        return evaluate_loss(raw, validation_loader, device, args.split_seed + 10000 + rank, dist)
    if validation_loader is not None and args.model == 'hamlet' and start_step == 0:
        baseline = {'step': 0, **validate()}
        if rank == 0:
            (args.output / 'baseline_validation.json').write_text(json.dumps(baseline, indent=2))
            with (args.output / 'validation.jsonl').open('a') as out:
                out.write(json.dumps(baseline)+'\n')
            if args.init_from:
                record_best(args.output, args.init_from, 0, baseline, baseline=True)
            print(json.dumps(baseline), flush=True)
    iterator = iter(loader)
    records = []
    begin = time.time()
    first_name, first_param, initial = None, None, None
    interval_loss, interval_steps = 0., 0
    for step in range(start_step + 1, run_end + 1):
        torch.cuda.synchronize(device)
        step_begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        valid_counts = []
        for micro in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch = move(batch, device)
            valid_counts.extend((~batch['action_is_pad']).sum(1).tolist())
            sync_ctx = model.no_sync() if (args.ddp and micro < args.grad_accum - 1) else nullcontext()
            with sync_ctx:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(batch)
                assert torch.isfinite(loss), f'Nonfinite loss: {loss}'
                total += loss.item() / args.grad_accum
                (loss / args.grad_accum).backward()
        if dist is not None:
            stats = torch.tensor([total], device=device)
            dist.all_reduce(stats)
            total = (stats / world).item()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        assert norm > 0, 'No gradient signal'
        if args.smoke and initial is None:
            first_name, first_param = next((k, v) for k, v in raw.named_parameters()
                                           if v.grad is not None and torch.count_nonzero(v.grad).item() > 0)
            initial = first_param.detach().clone()
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_begin
        interval_loss += total
        interval_steps += 1
        record = {'step': step, 'loss': total, 'train_loss': total,
                  'grad_norm': norm.item(), 'lr': optimizer.param_groups[0]['lr'],
                  'valid_action_steps': valid_counts, 'max_gpu_gb': torch.cuda.max_memory_allocated() / 1e9,
                  'elapsed_seconds': time.time() - begin, 'train_step_seconds': step_seconds}
        evaluating = validation_loader is not None and (step == 1 or step % args.eval_every == 0 or step == run_end)
        if evaluating and args.model == 'hamlet' and step == 1:
            evaluating = False  # Step-zero baseline already measured the initialized model.
        if evaluating:
            record.update(validate())
            record['train_loss_mean'] = interval_loss / interval_steps
            interval_loss, interval_steps = 0., 0
            if rank == 0:
                with (args.output / 'validation.jsonl').open('a') as out:
                    out.write(json.dumps(record) + '\n')
        if rank == 0 and (args.smoke or step % 10 == 0 or step == run_end or evaluating):
            print(json.dumps(record), flush=True)
            with (args.output / 'metrics.jsonl').open('a') as out:
                out.write(json.dumps(record) + '\n')
        records.append(record)
        if step % args.save_every == 0 or step == run_end:
            checkpoint_started = time.time()
            optimizer.zero_grad(set_to_none=True)
            if args.sharded_checkpoint:
                from checkpoint_shards import save_sharded
                torch.cuda.empty_cache()
                checkpoint = args.output / f'checkpoint-{step:06d}'
                save_sharded(raw, optimizer, scheduler, step, checkpoint, config, data.audit, ROOT, device, dist)
            else:
                local_rng = capture_rng(device)
                rank_rng_states = [None] * world if rank == 0 else None
                if dist is not None:
                    dist.gather_object(local_rng, rank_rng_states, dst=0)
                else:
                    rank_rng_states = [local_rng]
                if args.zero_optimizer:
                    torch.cuda.empty_cache()
                    optimizer.consolidate_state_dict(to=0)
                if rank == 0:
                    checkpoint = save_checkpoint(raw, optimizer, scheduler, step, args.output, config, data.audit, rank_rng_states)
                if dist is not None:
                    dist.barrier()
            if rank == 0:
                if args.model == 'hamlet' and evaluating:
                    record_best(args.output, checkpoint, step, record)
                checkpoint_record = dict(step=step, checkpoint=str(checkpoint), checkpoint_seconds=time.time()-checkpoint_started)
                with (args.output / 'checkpoint_timings.jsonl').open('a') as out:
                    out.write(json.dumps(checkpoint_record)+'\n')
                print(json.dumps(checkpoint_record), flush=True)
    if args.smoke:
        assert not torch.equal(initial, first_param), f'Parameter did not update: {first_name}'
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.manual_seed(123)
            before = model(batch).item()
        saved = load_file(str(checkpoint / 'trainable.safetensors'))
        with torch.no_grad():
            first_param.add_(.125)
            for k, value in raw.named_parameters():
                if value.requires_grad:
                    value.copy_(saved[k])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.manual_seed(123)
            after = model(batch).item()
        assert math.isfinite(after) and abs(before - after) < 1e-5
        training = torch.load(checkpoint / 'trainer.pt', map_location='cpu', weights_only=False)
        # Releasing the old moments avoids holding two AdamW states on a 32GB GPU.
        optimizer.state.clear()
        optimizer.load_state_dict(training['optimizer'])
        scheduler.load_state_dict(training['scheduler'])
        report = {'status': 'passed', 'model': args.model, 'optimizer_steps': args.steps,
                  'rtc_max_delay': rtc, 'lora': args.lora,
                  'real_dataset': data.audit['dataset'], 'boundary_checks': boundary,
                  'records': records, 'checkpoint': str(checkpoint), 'reloaded_loss': after,
                  'parameter_updated': first_name, 'pretrained_load': raw.load_report,
                  'limits': 'Training smoke test; no convergence or real-robot rollout claim'}
        (ROOT / 'reports' / f'smoke_{args.model}{args.report_suffix}.json').write_text(json.dumps(report, indent=2))
        print('SMOKE PASSED', args.model, args.report_suffix, flush=True)
    if dist is not None:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
