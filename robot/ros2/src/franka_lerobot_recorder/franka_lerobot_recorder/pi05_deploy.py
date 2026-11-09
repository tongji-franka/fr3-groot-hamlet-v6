"""PI0.5 project adapters for the existing asynchronous bounded FR3 executor."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def load_agent(project, checkpoint, device, *, dataset_root=None, task=None, inference_mode='async',
               rtc_execution_horizon=10, rtc_max_guidance_weight=10.,vision_precision='checkpoint',attention_backend=None):
    project = Path(project).expanduser().resolve(strict=True)
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if (checkpoint / 'pretrained_model').is_dir():
        checkpoint /= 'pretrained_model'
    spec = importlib.util.spec_from_file_location('fr3_pi05_inference', project / 'inference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extra = {}
    if vision_precision != 'checkpoint' or attention_backend is not None:
        config = json.loads((checkpoint/'config.json').read_text())
        if config.get('checkpoint_format') != 'fr3_pi05_frozen_v1':
            raise ValueError('Deployment compute options require the frozen-expert adapter')
        if vision_precision != 'checkpoint':extra['vision_precision'] = vision_precision
        if attention_backend is not None:extra['attention_backend'] = attention_backend
    agent = module.HockeyPI05(checkpoint, device=device, dataset_root=dataset_root, task=task,
                             inference_mode=inference_mode, rtc_execution_horizon=rtc_execution_horizon,
                             rtc_max_guidance_weight=rtc_max_guidance_weight, **extra)
    if hasattr(agent, 'model') and getattr(agent, 'metadata', {}).get('rtc_mode') == 'trained':
        from .inference_cache import install_frozen_pi05_cache
        install_frozen_pi05_cache(agent)
    agent.reset()
    return agent, checkpoint


class RTCInference:
    """Worker-only RTC inference, with observation age and measured inference latency."""
    def __init__(self, agent, fps, speed_scale, max_age, warmup_latency, journal):
        from lerobot.policies.rtc.latency_tracker import LatencyTracker
        self.agent, self.fps, self.speed_scale = agent, fps, speed_scale
        self.max_age, self.journal = max_age, journal
        self.latency = LatencyTracker()
        self.latency.add(warmup_latency)

    def warmup(self, observation, execution_horizon, input_age=0.):
        """Initialize thread-local CUDA paths before any bridge connection or goals."""
        previous = self.agent.predict_chunk(*observation)
        # First guided call initializes autograd/CUDA in this same worker. Measure
        # a subsequent fresh-vision call, without hiding any live inference delays.
        for _ in range(2):
            if hasattr(self.agent, 'clear_visual_cache'):
                self.agent.clear_visual_cache()
            started = time.monotonic()
            self.agent.predict_chunk(*observation, rtc_context=dict(
                previous_actions=previous, inference_delay=0, execution_horizon=execution_horizon))
            latency = time.monotonic() - started
        if hasattr(self.agent, 'reset'):
            self.agent.reset()
        self.latency.reset()
        self.latency.add(latency)
        self.journal.event('rtc_worker_warmup', inference_s=latency, contacts_robot=False)
        delay = math.ceil((input_age + latency + 1/self.fps)*self.fps*self.speed_scale)
        if delay >= execution_horizon:
            raise RuntimeError(f'Warmed RTC worker needs {delay} prefix steps in horizon {execution_horizon}; '
                               'cannot start before a usable overlap is available')

    def __call__(self, packet, freshness_check):
        observation, context = packet
        prefix = context['previous_actions']
        started = time.monotonic()
        kwargs = {}
        delay = horizon = None
        if prefix is not None:
            from franka_lerobot_recorder.deployment_safety import require_fresh
            require_fresh(context['previous_captured'], self.max_age, now=started, label='RTC prefix')
            # Convert physical seconds/ticks to the speed-scaled model action grid.
            elapsed = max(0., started-context.get('origin_stamp', context['state_stamp']))
            delay = math.ceil((elapsed+self.latency.max()+1/self.fps)*self.fps*self.speed_scale)
            horizon = min(self.agent.policy.config.rtc_config.execution_horizon, len(prefix))
            if delay >= horizon:
                raise RuntimeError(f'RTC inference delay {delay} leaves no overlap in horizon {horizon}')
            kwargs['rtc_context'] = dict(previous_actions=prefix, inference_delay=delay,
                                         execution_horizon=horizon)
        actions = self.agent.predict_chunk(*observation, freshness_check=freshness_check, **kwargs)
        finished = time.monotonic()
        latency = finished-started
        self.latency.add(latency)
        if prefix is not None:
            actual_delay = math.ceil(max(0., finished-context.get('origin_stamp', context['state_stamp']))
                                     *self.fps*self.speed_scale)
            if actual_delay > delay:
                self.journal.event('rtc_budget_exceeded', predicted_delay_steps=delay,
                                   actual_delay_steps=actual_delay, inference_s=latency)
                raise RuntimeError(f'RTC inference exceeded frozen prefix: {actual_delay} > {delay} steps')
        self.journal.event('rtc_inference', inference_delay_steps=delay, execution_horizon=horizon,
                           previous_prefix_length=0 if prefix is None else len(prefix), inference_s=latency)
        return actions


def validate_live_config(agent, config, allow_camera_replacement=()):
    if config['arm_joint_names'] != agent.metadata['joint_names']:
        raise ValueError('Live joint order differs from training')
    replacements = set(allow_camera_replacement)
    if not replacements <= set(agent.metadata['cameras']):
        raise ValueError('Camera replacement names must belong to the checkpoint cameras')
    for key, recorded in agent.metadata['cameras'].items():
        if key not in config['cameras']:
            raise ValueError(f'Missing live camera {key}; use the matching Teach camera configuration')
        live = config['cameras'][key]
        fields = ('model', 'width', 'height', 'fps', 'rotate_180')
        if key not in replacements:
            fields = ('serial', *fields)
        elif live['serial'] != recorded['serial']:
            print(f'Camera replacement: {key} serial {recorded["serial"]} -> {live["serial"]}; '
                  'verify the replacement has the training viewpoint and alignment', flush=True)
        for field in fields:
            if live[field] != recorded[field]:
                raise ValueError(f'Live camera {key}.{field} differs from training: {live[field]} != {recorded[field]}')


def offline_check(agent, checkpoint, max_start_distance=1.0, max_inference_age=1.2):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from franka_lerobot_recorder.act_deploy import validate_targets, target_distance_report

    root = Path(agent.metadata['dataset_root'])
    dataset = LeRobotDataset(f'local/{root.name}', root=root, video_backend='pyav')
    frame = dataset[0]
    q = frame['observation.state'].numpy()
    images = [(frame[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
              for key in agent.camera_keys]
    elapsed = []
    rtc = getattr(agent, 'inference_mode', 'async') == 'rtc'
    rtc_mode = getattr(getattr(agent.policy.config, 'rtc_config', None), 'mode', 'guided') if rtc else None
    prefix_errors = []
    previous = None
    for _ in range(4):
        start = time.monotonic()
        kwargs = {}
        if rtc and previous is not None:
            prefix = previous[agent.policy.config.n_action_steps:]
            horizon = min(agent.policy.config.rtc_config.execution_horizon, len(prefix))
            delay = min(agent.policy.config.rtc_training_max_delay, horizon-1) if rtc_mode == 'trained' else 0
            kwargs['rtc_context'] = dict(previous_actions=prefix, inference_delay=delay, execution_horizon=horizon)
        chunk = validate_targets(agent.predict_chunk(q, *images, **kwargs))
        if rtc_mode == 'trained' and previous is not None and delay:
            error = float(np.max(np.abs(chunk[:delay]-prefix[:delay])))
            if error > 2e-6:
                raise ValueError(f'Trained RTC changed a committed prefix by {error} rad')
            prefix_errors.append(error)
        previous = chunk
        elapsed.append(time.monotonic() - start)
    report = dict(policy='pi05', checkpoint=str(checkpoint), task=agent.metadata['task'],
                  chunk_shape=list(chunk.shape), first_inference_s=elapsed[0],
                  subsequent_inference_s=elapsed[1:], finite=True, contacts_robot=False,
                  inference_mode='rtc' if rtc else 'async', rtc_mode=rtc_mode,
                  vision_precision=agent.metadata.get('vision_precision'),
                  attention_backend=agent.metadata.get('attention_backend'),
                  rtc_guided_calls=3 if rtc and rtc_mode == 'guided' else 0,
                  rtc_trained_calls=3 if rtc and rtc_mode == 'trained' else 0,
                  rtc_prefix_max_errors_rad=prefix_errors,
                  first_dataset_frame_start_checks=[target_distance_report(q, chunk[i], False, i, max_start_distance)
                                                    for i in range(min(3, len(chunk)))])
    from .deployment_safety import continuous_inference_budget
    report['continuous_budget_estimate'] = continuous_inference_budget(
        max(elapsed[1:]), .05, max_inference_age, 30, pipelined=not rtc)
    report['continuous_budget_estimate']['input_age_assumption'] = '50 ms; live startup measures the actual camera age'
    print(json.dumps(report, indent=2))
    return report


def main(args=None):
    from franka_lerobot_recorder.act_deploy import main as deploy

    return deploy(args, policy='pi05')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; stop requested', file=sys.stderr)
        sys.exit(130)
