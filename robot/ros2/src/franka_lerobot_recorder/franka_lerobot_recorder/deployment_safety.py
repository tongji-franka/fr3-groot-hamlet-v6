"""Device-free timing, admission and durable evidence for FR3 deployment."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import uuid

import numpy as np

from .observation_timing import ObservationNotReady


def require_fresh(captured, max_age, *, now=None, label='Observation'):
    now = time.monotonic() if now is None else now
    age = now - captured
    if not math.isfinite(age) or not 0 <= age <= max_age:
        raise RuntimeError(f'{label} is stale or has an invalid timestamp: age={age}, limit={max_age}')
    return age


def continuous_inference_budget(latency, input_age, max_age, fps, refresh_steps=1, pipelined=False):
    """Two calls and handoff; async prefetch removes the intermediate result poll."""
    if (any(not math.isfinite(x) for x in (latency, input_age, max_age, fps))
            or latency < 0 or input_age < 0 or max_age <= 0 or fps <= 0
            or type(refresh_steps) is not int or refresh_steps < 1):
        raise ValueError('Invalid continuous inference budget inputs')
    required = input_age + 2*latency + (refresh_steps+1+(not pipelined))/fps
    return dict(fits=required < max_age, policy_latency_s=latency, input_age_s=input_age,
                required_age_s=required, max_prediction_age_s=max_age,
                margin_s=max_age-required, refresh_steps=refresh_steps,
                pipelined=pipelined,
                note='Measured startup estimate; runtime freshness and overrun guards remain authoritative')


class ControlSlotExpired(RuntimeError):
    """A target's time slot passed before dispatch; it must not be replayed."""


class ControlClock:
    def __init__(self, fps, epoch=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be positive and finite')
        self.period = 1 / fps
        self.epoch = time.monotonic() if epoch is None else epoch
        self.last_step = -1
        self.last_sent = None

    def due(self):
        due = self.epoch + (self.last_step + 1) * self.period
        return due if self.last_sent is None else max(due, self.last_sent + self.period)

    def index(self, now):
        return math.floor((now-self.epoch) / self.period + 1e-7)

    def sent(self, started, step):
        if step <= self.last_step or (self.last_sent is not None and started < self.last_sent+self.period-1e-7):
            raise RuntimeError('Control slot replay or compressed send interval')
        skipped = step-self.last_step-1
        self.last_step, self.last_sent = step, started
        return skipped

    def skip(self, step):
        if step <= self.last_step:
            raise RuntimeError('Control slot replay')
        skipped = step-self.last_step
        self.last_step = step
        return skipped


@dataclass(frozen=True)
class Observation:
    sequence: int
    state_stamp: float
    captured: float
    finished: float
    state: np.ndarray
    frames: tuple
    timing: dict = None


class ObservationPump:
    """Only the producer copies images; latest() never waits for camera IO."""
    def __init__(self, reader, fps):
        self.reader, self.period = reader, 1/fps
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.refresh_requested = threading.Event()
        self.version = 0
        self.value = self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name='fr3-observation')
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                self.refresh_requested.clear()
                started = time.monotonic()
                try:
                    value = self.reader()
                except ObservationNotReady:
                    if self.value is not None:
                        raise
                    self.stop.wait(self.period)
                    continue
                with self.condition:
                    self.value = value
                    self.version += 1
                    self.condition.notify_all()
                self.refresh_requested.wait(max(0., self.period-(time.monotonic()-started)))
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def latest(self, wait=False):
        with self.condition:
            if wait:
                self.condition.wait_for(lambda: self.value is not None or self.error is not None, timeout=5)
            if self.error is not None:
                if isinstance(self.error, KeyboardInterrupt):
                    raise KeyboardInterrupt from self.error
                raise RuntimeError('Observation worker failed') from self.error
            if self.value is None:
                raise RuntimeError('No complete observation snapshot')
            return self.value

    def refresh(self, timeout=.1):
        """Worker-only request; camera IO stays on the observation producer."""
        with self.condition:
            version = self.version
            self.refresh_requested.set()
            ready = self.condition.wait_for(
                lambda:self.version>version or self.error is not None or self.stop.is_set(),timeout=timeout)
            if not ready:
                raise RuntimeError('Fresh observation request timed out')
        return self.latest()

    def close(self):
        self.stop.set()
        self.refresh_requested.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('Observation worker did not stop')


class RawActionGuard:
    """Measure unmodified model outputs, independently of velocity limiting."""
    def __init__(self, step_limits=None, handoff_limits=None):
        self.step_limits = self._limits(step_limits)
        self.handoff_limits = self._limits(handoff_limits)

    @staticmethod
    def _limits(value):
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError('Raw action limits must contain seven positive finite radians')
        return value

    def check_chunk(self, actions):
        peak = np.max(np.abs(np.diff(actions, axis=0)), axis=0) if len(actions)>1 else np.zeros(7)
        if self.step_limits is not None and (peak > self.step_limits).any():
            raise RuntimeError(f'Unsafe raw model step: {peak.tolist()}')
        return peak.tolist()

    def check_handoff(self, old, new):
        delta = np.abs(new-old)
        if self.handoff_limits is not None and (delta > self.handoff_limits).any():
            raise RuntimeError(f'Unsafe raw chunk handoff: {delta.tolist()}')
        return delta.tolist()


def atomic_json(path, data):
    path = Path(path)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Journal:
    """Unique run, durable incremental events, bounded queue with explicit failure."""
    def __init__(self, root, metadata, capacity=4096):
        self.path = Path(root) / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+uuid.uuid4().hex[:12])
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, status='running', success='unknown', started_wall=time.time())
        atomic_json(self.path/'run.json', self.metadata)
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.closed = False
        self.stream = (self.path/'events.jsonl').open('a', buffering=1)
        self.thread = threading.Thread(target=self._run, daemon=True, name='fr3-journal')
        self.thread.start()

    def _append(self, payload):
        self.stream.write(payload)
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def _run(self):
        while True:
            payload = self.queue.get()
            try:
                if payload is None:
                    return
                # Even after producer overflow, drain already admitted events.
                self._append(payload)
            except BaseException as error:
                self.error = error
            finally:
                self.queue.task_done()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Deployment recording failed') from self.error
        if self.closed:
            raise RuntimeError('Deployment journal is closed')

    def event(self, event, **data):
        self.check()
        payload = json.dumps(dict(event=event, monotonic=time.monotonic(), **data), allow_nan=False)+'\n'
        try:
            self.queue.put_nowait(payload)
        except queue.Full as error:
            self.error = error
            raise RuntimeError('Deployment journal queue full; stopping') from error

    def close(self, status, error=None, **summary):
        if self.closed:
            return
        try:
            self.queue.put(None, timeout=5)
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError('Deployment journal flush timed out')
            self.stream.close()
            summary['measurements'] = summarize_events(self.path/'events.jsonl')
            self.metadata.update(summary, status='recording_error' if self.error else status,
                                 error=error, recording_error=repr(self.error) if self.error else None,
                                 ended_wall=time.time())
            atomic_json(self.path/'run.json', self.metadata)
            self.check()
        finally:
            self.closed = True


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path, policy, dataset_root=None):
    root = Path(path).expanduser().resolve(strict=True)
    if (root/'pretrained_model').is_dir():
        root /= 'pretrained_model'
    config = json.loads((root/'config.json').read_text())
    if policy == 'pi05' and config.get('checkpoint_format') == 'fr3_pi05_frozen_v1':
        from fr3_pi05.inference import checkpoint_identity as frozen_identity
        return frozen_identity(root, dataset_root)
    if config.get('type') != policy:
        raise ValueError('Checkpoint policy type mismatch')
    if config.get('n_obs_steps') != 1:
        raise ValueError('Deployment requires current-frame-only n_obs_steps=1')
    if (policy == 'act' and config.get('temporal_ensemble_coeff') is not None
            or policy == 'pi05' and config.get('rtc_config') is not None):
        raise ValueError('Chunk runner does not support temporal ensembling or RTC')
    for group, key in [('input_features', 'observation.state'), ('output_features', 'action')]:
        if config.get(group, {}).get(key, {}).get('shape') != [7]:
            raise ValueError('Checkpoint requires seven state/action dimensions')
    chunk = config.get('chunk_size')
    cadence = config.get('n_action_steps')
    if type(chunk) is not int or type(cadence) is not int or not 1 <= cadence <= chunk:
        raise ValueError('Invalid checkpoint chunk_size/n_action_steps')
    paths = {str(p.relative_to(root)): p for p in root.rglob('*')
             if p.is_file() and p.suffix in ('.json', '.safetensors')}
    required = 'adapter_model.safetensors' if policy == 'pi05' else 'model.safetensors'
    if required not in paths:
        raise ValueError(f'Missing checkpoint {required}')
    joint_names = [f'fr3_joint{i}' for i in range(1, 8)]
    if policy == 'pi05':
        adapter = json.loads((root/'adapter_config.json').read_text())
        base = Path(adapter['base_model_name_or_path']).expanduser().resolve(strict=True)
        paths['base/model.safetensors'] = base/'model.safetensors'
        paths['base/config.json'] = base/'config.json'
        training = json.loads((root/'train_config.json').read_text())
        dataset = Path(dataset_root or training['dataset']['root']).expanduser().resolve(strict=True)
        paths['dataset/contract.json'] = dataset/'recording/contract.json'
        contract = json.loads(paths['dataset/contract.json'].read_text())
        units_valid = contract['units']['arm_position'] == 'rad'
        order_valid = all(contract['features'][key]['names'] == joint_names
                          for key in ('observation.state', 'action'))
    else:
        contract = json.loads((root/'hockey_preprocessing.json').read_text())
        units_valid = contract['state_units'] == contract['action_units'] == 'rad'
        order_valid = contract['joint_names'] == joint_names
    if (contract['fps'] != 30 or not units_valid or not order_valid
            or contract['action_definition'] not in ('next_sample_joint_position_rad',
                                                     'next_uniform_time_interpolated_joint_position_rad')):
        raise ValueError('Checkpoint must declare FR3 30 Hz J1-J7 absolute-radian actions')
    files = {name: file_digest(p) for name, p in sorted(paths.items())}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    step = root.parent.name if root.name == 'pretrained_model' else root.name
    return dict(checkpoint=str(root), checkpoint_step_from_path=int(step) if step.isdigit() else None,
                policy=policy, chunk_size=chunk, n_action_steps=cadence,
                action_definition=contract['action_definition'], joint_names=joint_names,
                sha256=identity, files=files)


def load_safety_profile(path, checkpoint_sha256, motion_profile_id):
    profile = json.loads(Path(path).read_text())
    if (profile.get('checkpoint_sha256') != checkpoint_sha256
            or profile.get('motion_profile_id') != motion_profile_id
            or not isinstance(profile.get('evidence'), str) or not profile['evidence'].strip()):
        raise ValueError('Safety profile must match checkpoint/motion profile and identify measurement evidence')
    RawActionGuard(profile['max_raw_step_rad'], profile['max_handoff_rad'])
    return profile


def summarize_events(path):
    """Offline/after-stop analysis; observed maxima are NOT automatically safety limits."""
    metrics = {name: [] for name in ('send_period_s', 'send_duration_s', 'inference_age_s', 'snapshot_age_s')}
    stages = {
        'observation': ('camera_read_s', 'observation_build_s', 'camera_capture_skew_s',
                        'camera_alignment_error_s', 'state_interpolation_gap_s', 'state_transport_s',
                        'state_callback_to_publish_s', 'state_age_at_observation_s'),
        'inference_finished': ('input_queue_s', 'input_age_at_worker_start_s', 'policy_execution_s',
                               'before_last_input_check_s', 'after_last_input_check_s', 'capture_to_prediction_ready_s'),
        'send_dispatched': ('capture_to_send_s', 'result_wait_s', 'inference_finish_to_send_s'),
    }
    dispatch_times, native_times, invalid_timing = {}, {}, {}
    replaced_requests = 0
    def measure(key, value):
        if value is None:
            return
        value = float(value)
        if not math.isfinite(value) or value < 0:
            invalid_timing[key] = invalid_timing.get(key, 0)+1
            return
        metrics.setdefault(key, []).append(value)
    raw_peak, handoff_peak, tracking_peak = np.zeros(7), np.zeros(7), np.zeros(7)
    dispatched, accepted = set(), set()
    skipped = 0
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # A killed writer may leave an incomplete final JSONL record.
            event = json.loads(line)
            for key in ('send_period_s', 'send_duration_s', 'inference_age_s', 'snapshot_age_s'):
                value = event.get(key)
                measure(key, value)
            for key in stages.get(event['event'], ()):
                measure(key, event.get(key))
            if event['event'] == 'observation':
                for camera, stamp in event.get('cameras', {}).items():
                    measure(f'camera/{camera}/capture_to_receive_s',
                            stamp['received_monotonic']-stamp['captured_monotonic'])
                    measure(f'camera/{camera}/host_clock_read_bound_s', stamp.get('clock_error_bound_s'))
            if event['event'] == 'inference_request_replaced':
                replaced_requests += 1
            if event['event'] == 'prediction':
                actions = np.asarray(event['actions'])
                if len(actions)>1:
                    raw_peak = np.maximum(raw_peak, np.abs(np.diff(actions, axis=0)).max(0))
            delta = event.get('aligned_handoff_delta_rad')
            if delta is not None:
                handoff_peak = np.maximum(handoff_peak, delta)
            delta = event.get('sdk_tracking_delta_rad')
            if delta is not None:
                tracking_peak = np.maximum(tracking_peak, np.abs(delta))
            if event['event'] == 'send_dispatched':
                dispatched.add(event['sequence'])
                skipped += event.get('skipped_slots', 0)
                if 'captured_monotonic' in event:
                    dispatch_times[event['sequence']] = (event['captured_monotonic'], event['send_started_monotonic'],
                                                        event.get('prediction_id'))
            if event['event'] == 'native_acceptance':
                accepted.add(event['sequence'])
                native_times[event['sequence']] = event
            if event.get('accepted_sequence', 0)>0:
                accepted.add(event['accepted_sequence'])
    # Native acknowledgments can be enqueued before the Python send-completion event.
    # Correlate by command sequence after reading the stream, never by log line order.
    seen_predictions = set()
    for sequence, sent in sorted(dispatch_times.items()):
        ack = native_times.get(sequence)
        if ack is None:
            continue
        captured, send_start, prediction_id = sent
        rx, admit, output = (ack[key] for key in ('native_received_monotonic', 'native_accepted_monotonic', 'native_output_monotonic'))
        measure('capture_to_native_accept_s', admit-captured)
        measure('capture_to_native_output_s', output-captured)
        if prediction_id is not None and prediction_id not in seen_predictions:
            measure('capture_to_first_native_output_s', output-captured)
            seen_predictions.add(prediction_id)
        measure('send_to_native_receive_s', rx-send_start)
        measure('native_receive_to_accept_s', admit-rx)
        measure('native_accept_to_output_s', output-admit)
        measure('native_output_to_ack_s', ack['ack_received_monotonic']-output)
        measure('state_publish_to_ack_s', ack['ack_received_monotonic']-ack['native_published_monotonic'])
    return dict(
        timing={key: dict(zip(('p50', 'p95', 'p99', 'max'), map(float, np.percentile(values, [50, 95, 99, 100]))))
                for key, values in metrics.items() if values},
        raw_step_peak_rad=raw_peak.tolist(), aligned_handoff_peak_rad=handoff_peak.tolist(),
        sdk_tracking_peak_rad=tracking_peak.tolist(), skipped_slots=skipped,
        dispatched_count=len(dispatched), observed_accepted_count=len(accepted),
        native_timing_correlated_count=len(dispatch_times.keys() & native_times.keys()),
        overwritten_inference_requests=replaced_requests, invalid_timing_samples=invalid_timing,
        acceptance_not_observed=sorted(dispatched-accepted),
        caveat='Capture-to-native-output ends at command generation, not physical attainment. '
               'Robot state alignment uses native callback receipt times; raw robot times are retained. '
               'Host clock bounds do not bound the SDK device-clock fit. '
               'Policy execution includes preprocessing, GPU completion and postprocessing, not only GPU kernels. '
               'inference_age_s includes observation age and worker/preprocessing delay. '
               'Measured peaks are not approved safety thresholds.')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze or manually label an existing FR3 deployment run')
    parser.add_argument('run', type=Path)
    parser.add_argument('--outcome', choices=('success', 'failure', 'unknown'))
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    summary_path = args.run/'run.json'
    summary = json.loads(summary_path.read_text())
    if args.outcome is not None:
        if summary['status'] == 'running':
            parser.error('Refuse to label an active/unfinalized run; inspect and recover its status first')
        summary.update(success=args.outcome, operator_note=args.note, labelled_wall=time.time())
        atomic_json(summary_path, summary)
    report = summarize_events(args.run/'events.jsonl')
    atomic_json(args.run/'measurements.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
