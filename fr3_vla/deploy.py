"""Deploy an exported v4 checkpoint through the existing bounded FR3 executor."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from .contract import (DATASET, FR3, MODELS, ROOT, digest, read_dataset_contract,
                       read_manifest, validate_live_config, write_json)


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--training-root', type=Path, help='Relocated cloud assets, upstream sources and native/legacy environments')
    parser.add_argument('--dataset-root', type=Path, default=DATASET)
    parser.add_argument('--fr3-root', type=Path, default=FR3)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/fr3_teach_v4.yaml')
    parser.add_argument('--device', default='cuda')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-config', action='store_true')
    modes.add_argument('--offline-check', action='store_true')
    modes.add_argument('--observe-only', action='store_true')
    modes.add_argument('--move', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Explicitly permit a real native bridge connection')
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--duration', type=float, default=10.)
    parser.add_argument('--speed-scale', type=float, default=1.0)
    parser.add_argument('--max-inference-age', type=float, default=1.2,
                        help='Prediction age limit: up to 1.2 s (default: 1.2 s)')
    parser.add_argument('--lock-first-joint', action='store_true', help='Hold J1 at its initial measured position')
    parser.add_argument('--lock-fifth-joint', action='store_true', help='Hold J5 at its initial measured position')
    parser.add_argument('--max-target-distance', type=float, default=1.0,
                        help='Running per-joint target-distance limit (0, 1] rad; default 1.0; match bridge')
    parser.add_argument('--max-start-distance', type=float, default=1.)
    parser.add_argument('--bridge-startup-timeout', type=float, default=60.)
    parser.add_argument('--actions-per-chunk', type=int)
    parser.add_argument('--handoff-steps', type=int, choices=range(1, 9), default=None)
    parser.add_argument('--hamlet-kv-cache', action=argparse.BooleanOptionalAction, default=None,
                        help='Reuse action-head cross-attention K/V within one HAMLET denoising call (default: on for HAMLET)')
    parser.add_argument('--hamlet-bf16-linear', action=argparse.BooleanOptionalAction, default=None,
                        help='Keep HAMLET action-head linear weights in their BF16 autocast format (default: on)')
    parser.add_argument('--hamlet-image-cache-size', type=int, default=None,
                        help='HAMLET preprocessed image cache capacity; 0 disables it (default: 96)')
    parser.add_argument('--hamlet-cuda-graph', action=argparse.BooleanOptionalAction, default=None,
                        help='Capture the HAMLET action sampler during warmup (default: on; CUDA only)')
    parser.add_argument('--safety-profile', type=Path)
    parser.add_argument('--diagnostic-output', type=Path)
    parser.add_argument('--handoff-request', type=Path, help='Opt-in manual handoff request file; used by the demo supervisor')
    parser.add_argument('--stage-status', type=Path, help='Opt-in atomic stage status for the demo supervisor')
    options = parser.parse_args(argv)
    if options.handoff_request or options.stage_status:
        if not (options.handoff_request and options.stage_status and options.move and options.execute and options.duration == 0):
            parser.error('Manual handoff requires both IPC paths, --move --execute and --duration 0')
        if options.handoff_request.exists() or options.stage_status.exists():
            parser.error('Manual handoff IPC paths must be new for each run')
        ipc = [options.handoff_request.resolve(), options.stage_status.resolve()]
        if len(set(ipc)) != 2 or (options.diagnostic_output and options.diagnostic_output.resolve() in ipc):
            parser.error('Manual handoff request, status and diagnostic paths must be distinct')
    if options.hamlet_kv_cache is None:
        options.hamlet_kv_cache = options.model == 'hamlet'
    if options.hamlet_kv_cache and options.model != 'hamlet':
        parser.error('--hamlet-kv-cache requires model hamlet')
    for name in ('hamlet_bf16_linear', 'hamlet_cuda_graph'):
        value = getattr(options, name)
        if value is None:
            setattr(options, name, options.model == 'hamlet')
        elif value and options.model != 'hamlet':
            parser.error(f'--{name.replace("_", "-")} requires model hamlet')
    if options.hamlet_image_cache_size is None:
        options.hamlet_image_cache_size = 96 if options.model == 'hamlet' else None
    elif options.model != 'hamlet' or options.hamlet_image_cache_size < 0:
        parser.error('--hamlet-image-cache-size requires HAMLET and a nonnegative integer')
    if (options.lock_first_joint or options.lock_fifth_joint) and (options.offline_check or options.check_config):
        parser.error('Joint holds require live robot state; use --observe-only or --move')
    if options.max_target_distance is not None and (not math.isfinite(options.max_target_distance) or not 0 < options.max_target_distance <= 1):
        parser.error('max-target-distance must be finite and within (0, 1] rad')
    if options.handoff_steps is None:
        options.handoff_steps = 8 if options.model == 'hamlet' else 5
    if not options.check_config and options.checkpoint is None:
        parser.error('--checkpoint is required except for --check-config')
    if options.move and not options.execute:
        parser.error('--move requires --execute')
    if options.execute and (options.offline_check or options.check_config):
        parser.error('Offline/config checks cannot use --execute')
    for key, low, high in [('speed_scale', .1, 1), ('max_inference_age', .001, 1.2),
                           ('max_start_distance', .001, 1), ('duration', 0, math.inf),
                           ('bridge_startup_timeout', .001, 300)]:
        value = getattr(options, key)
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(f'Invalid --{key.replace("_", "-")}')
    if not 1024 <= options.port <= 65535:
        parser.error('Port must be within 1024..65535')
    options.observe_only = not options.move
    options.policy_name = options.model
    options.diagnostic_output = options.diagnostic_output or ROOT / f'models/{options.model}/artifacts/last-diagnostic.json'
    return options


def main(argv=None):
    options = parse_options(argv)
    read_dataset_contract(options.dataset_root)
    import yaml
    config = yaml.safe_load(options.config.read_text())
    if options.checkpoint is None:
        manifest = json.loads((ROOT / f'models/{options.model}/configs/fr3_deployment.template.json').read_text())
    else:
        # Agent verifies all weight bytes before loading; config-only checks do it here.
        manifest = read_manifest(options.checkpoint, options.model, options.dataset_root,
                                 verify_weights=options.check_config, training_root=options.training_root)
    validate_live_config(manifest, config)
    if options.check_config:
        report = dict(model=options.model, contacts_robot=False, loads_model=False,
                      checkpoint_validated=options.checkpoint is not None, task=manifest['task'],
                      horizon=manifest['horizon'], replan_steps=manifest['replan_steps'],
                      joint_units='absolute_rad', cameras=manifest['camera_order'],
                      checkpoint_format=manifest.get('checkpoint_format', 'legacy_full_export'),
                      inference=manifest.get('inference'))
        if options.model == 'hamlet':
            report.update({name: getattr(options, name) for name in (
                'hamlet_kv_cache', 'hamlet_bf16_linear', 'hamlet_image_cache_size', 'hamlet_cuda_graph')})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    from .cloud import FORMAT
    cloud_runtime = manifest.get('checkpoint_format') == FORMAT
    if cloud_runtime and options.move:
        from .cloud import require_motion_parity
        require_motion_parity(options.checkpoint)
    if options.model == 'hamlet' and not cloud_runtime and manifest['replan_steps'] != 4:
        raise ValueError('HAMLET inference cadence must match training memory_stride=4')
    options.actions_per_chunk = options.actions_per_chunk or manifest['horizon']
    if not manifest['replan_steps'] <= options.actions_per_chunk <= manifest['horizon']:
        raise ValueError('Require replan_steps <= actions_per_chunk <= checkpoint horizon')
    sys.path.insert(0, str(options.fr3_root / 'ros2/src/franka_lerobot_recorder'))
    from franka_lerobot_recorder import act_deploy as executor
    from .adapters import Agent
    agent = Agent(options.model, options.checkpoint, options.dataset_root, options.device, options.training_root,
                  hamlet_kv_cache=options.hamlet_kv_cache, hamlet_bf16_linear=options.hamlet_bf16_linear,
                  hamlet_image_cache_size=options.hamlet_image_cache_size, hamlet_cuda_graph=options.hamlet_cuda_graph)
    if options.model == 'hamlet':
        print('HAMLET inference: '+json.dumps({name: agent.metadata[name] for name in (
            'hamlet_kv_cache', 'hamlet_bf16_linear', 'hamlet_image_cache_size', 'hamlet_cuda_graph')}), flush=True)
    if options.offline_check:
        from .data import V4Dataset
        dataset = V4Dataset(options.dataset_root)
        state, images, _ = dataset.sample(horizon=manifest['horizon'])
        if options.model == 'wsa' or cloud_runtime:
            agent.backend.history.observe(images, 0.)
        start = time.monotonic()
        chunk = agent.predict_chunk(state, *images)
        report = dict(model=options.model, contacts_robot=False, shape=list(chunk.shape),
            inference_seconds=time.monotonic() - start,
            first_target=executor.target_distance_report(state, chunk[0], False, 0, options.max_start_distance))
        if options.model == 'hamlet':
            report.update({name: agent.metadata[name] for name in (
                'hamlet_kv_cache', 'hamlet_bf16_linear', 'hamlet_image_cache_size', 'hamlet_cuda_graph')})
            if agent.action_graph is not None:
                report['hamlet_cuda_graph_state'] = agent.action_graph.stats()
        write_json(options.diagnostic_output, report)
        print(json.dumps(report, indent=2))
        return report
    options.checkpoint_identity = dict(policy=options.model, checkpoint=str(options.checkpoint.resolve()),
                                      sha256=digest(options.checkpoint / 'fr3_deployment.json'),
                                      files=manifest['weights_sha256'])
    if options.safety_profile:
        from franka_lerobot_recorder.deployment_safety import load_safety_profile
        options.safety_profile_data = load_safety_profile(options.safety_profile,
                    options.checkpoint_identity['sha256'], executor.PROFILE_ID)
    if options.model == 'wsa':
        from .wsa import run_with_history
        return run_with_history(executor, options, agent, config)
    if cloud_runtime:
        from .cloud import run_with_history
        return run_with_history(executor, options, agent, config)
    return executor.run(options, agent, config)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
