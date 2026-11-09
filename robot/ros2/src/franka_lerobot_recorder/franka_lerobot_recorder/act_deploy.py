"""Local asynchronous HockeyACT inference and bounded FR3 action streaming."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import struct
import sys
import threading
import time
import inspect
import signal

import numpy as np
from franka_lerobot_recorder.deployment_safety import (
    ControlClock, ControlSlotExpired, Observation, ObservationPump, RawActionGuard, Journal,
    require_fresh, atomic_json, checkpoint_identity, load_safety_profile, file_digest, continuous_inference_budget)
from franka_lerobot_recorder.observation_timing import StateHistory

COMMAND = struct.Struct('<17d')
STATE = struct.Struct('<41d')
PROTOCOL_VERSION = 6
COMMAND_WATCHDOG_SECONDS = .120
HANDOFF_POLICY_VERSION = "freshness-budget-v1"
HELLO, GOAL, STATUS, STOP = 32000, 32001, 32002, 32003
LOWER = np.array([-2.8007, -1.7361, -2.8007, -2.9770, -2.7763, 0.5398, -2.9508])
UPPER = np.array([2.8007, 1.7361, 2.8007, -0.2169, 2.7763, 4.5216, 2.9508])


def validate_collision_limits(collision):
    if not isinstance(collision, dict):
        raise ValueError('collision must be a mapping of phase-specific threshold arrays')
    for kind, size in [('torque', 7), ('force', 6)]:
        for phase in ('acceleration', 'nominal'):
            bounds = []
            for bound in ('lower', 'upper'):
                name = f'{bound}_{kind}_{phase}'
                values = collision.get(name)
                if (not isinstance(values, list) or len(values) != size
                        or any(type(x) not in (int, float) for x in values)):
                    raise ValueError(f'collision.{name} requires {size} numbers')
                values = np.asarray(values, dtype=np.float64)
                if not np.isfinite(values).all() or (values < 0).any() or (bound == 'upper' and (values <= 0).any()):
                    raise ValueError(f'Invalid collision.{name}; thresholds must be finite and nonnegative, upper > 0')
                bounds.append(values)
            if (bounds[0] > bounds[1]).any():
                raise ValueError(f'collision {kind}/{phase}: lower contact thresholds must not exceed upper collision thresholds')


def load_motion_limits():
    paths = [Path(__file__).resolve().parents[1] / 'config/act_limits.json',
             Path(sys.prefix) / 'share/franka_lerobot_recorder/config/act_limits.json']
    paths.extend(parent / 'share/franka_lerobot_recorder/config/act_limits.json'
                 for parent in Path(__file__).resolve().parents)
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise FileNotFoundError('Missing config/act_limits.json')
    data = path.read_bytes()
    limits = json.loads(data)
    validate_collision_limits(limits.get('collision'))
    for name, maximum in [('velocity', [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]),
                          ('acceleration', [10.] * 7), ('jerk', [5000.] * 7)]:
        values = np.asarray(limits[name], dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all() or (values <= 0).any() or (values >= maximum).any():
            raise ValueError(f'Invalid {name} profile; must be below FR3 nominal maxima')
    return limits, int(hashlib.sha256(data).hexdigest()[:8], 16)


MOTION_LIMITS, PROFILE_ID = load_motion_limits()
VELOCITY_LIMITS = np.asarray(MOTION_LIMITS['velocity'])
TARGET_DISTANCE = 1.25 * np.maximum(.1, .05 + .075 * VELOCITY_LIMITS)


def validate_speed_scale(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .1 <= value <= 1:
        raise ValueError('speed-scale must be finite and within [0.1, 1]')
    return float(value)


def validate_targets(actions):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError('Expected a nonempty [chunk_size, 7] action chunk')
    if not np.isfinite(actions).all() or (actions < LOWER).any() or (actions > UPPER).any():
        raise ValueError('ACT target outside finite FR3 joint limits with 0.1 rad margin')
    return actions


def validate_max_target_distance(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('max-target-distance must be finite and within (0, 1] rad')
    return float(value)


def target_distance_report(measured, target, started, step, max_start_distance=1.0, max_target_distance=None):
    max_target_distance = validate_max_target_distance(max_target_distance)
    if not math.isfinite(max_start_distance) or not 0 < max_start_distance <= 1:
        raise ValueError('max-start-distance must be finite and within (0, 1] rad')
    measured = np.asarray(measured, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if measured.shape != (7,) or target.shape != (7,) or not np.isfinite([measured, target]).all():
        raise ValueError('Target-distance check requires seven finite measured/target angles')
    running_limits = TARGET_DISTANCE if max_target_distance is None else np.full(7, max_target_distance)
    limits = running_limits if started else np.full(7, max_start_distance)
    delta = target - measured
    return dict(stage='running' if started else 'startup', control_step=step,
                measured_rad=measured.tolist(), target_rad=target.tolist(), delta_rad=delta.tolist(),
                limit_rad=limits.tolist(), max_abs_delta_rad=float(np.abs(delta).max()),
                exceeded_joints=[f'J{i+1}' for i in np.flatnonzero(np.abs(delta) > limits)])


class TargetDistanceError(RuntimeError):
    def __init__(self, report, policy_name='ACT'):
        details = '; '.join(
            f'J{i+1}: measured={report["measured_rad"][i]:.6f}, target={report["target_rad"][i]:.6f}, '
            f'delta={report["delta_rad"][i]:+.6f}, limit={report["limit_rad"][i]:.6f} rad'
            for i in range(7) if f'J{i+1}' in report['exceeded_joints'])
        super().__init__(f'{policy_name} target too far: stage={report["stage"]}, step={report["control_step"]}; {details}')


class ActionBuffer:
    """Actions retain observation-relative time; latency never shifts a chunk forward."""
    def __init__(self, fps, max_age=0.5, speed_scale=1.0, raw_guard=None, handoff_steps=5, locked_joints=None):
        if type(handoff_steps) is not int or not 1 <= handoff_steps <= 8:
            raise ValueError("handoff_steps must be an integer in [1, 8]")
        self.locked_joints = dict(locked_joints or {})
        for joint, value in self.locked_joints.items():
            if (type(joint) is not int or not 0 <= joint < 7
                    or isinstance(value, bool) or not isinstance(value, (int, float, np.floating))
                    or not math.isfinite(value) or not LOWER[joint] <= value <= UPPER[joint]):
                raise ValueError('Joint hold requires valid joint indices and finite positions inside joint limits')
        self.handoff_steps = handoff_steps
        self.fps, self.max_age = fps, max_age
        self.speed_scale = validate_speed_scale(speed_scale)
        self.velocity_limits = VELOCITY_LIMITS * self.speed_scale
        self.actions = {}
        self.latest_origin = None
        self.latest_captured = None
        self.raw_guard = raw_guard or RawActionGuard()
        self.measurements = {}
        self.trajectory = {}

    def merge(self, origin, captured, actions, step, now):
        actions = validate_targets(actions)
        require_fresh(captured, self.max_age, now=now, label='Inference result')
        if self.latest_origin is not None and origin <= self.latest_origin:
            return False
        peak = self.raw_guard.check_chunk(actions)
        if self.locked_joints:
            actions = actions.copy()
            for joint, value in self.locked_joints.items():
                actions[:, joint] = value
        # Dilate model time, then interpolate back to the unchanged physical control rate.
        count = math.ceil(len(actions) / self.speed_scale - 1e-9)
        future = {}
        for i in range(max(0, step-origin), count):
            index = min(i * self.speed_scale, len(actions)-1)
            left = int(index)
            fraction = index-left
            future[origin+i] = (1-fraction)*actions[left] + fraction*actions[min(left+1, len(actions)-1)]
        if not future:
            raise RuntimeError('Inference finished after its entire action chunk expired')
        handoff = None
        blend_steps = None
        previous_age_at_handoff_end = None
        if step in self.actions and step in future:
            handoff = self.raw_guard.check_handoff(self.actions[step], future[step])
            overlap = future.keys() & self.actions.keys()
            blend_steps = min(self.handoff_steps, max(overlap)-step)
            if blend_steps < 1:
                raise RuntimeError('Insufficient overlap for continuous chunk handoff')
            # Use the configured 1..8-tick C2 transition only while the old
            # prediction stays fresh through its end; never extend its age limit.
            while (blend_steps > 1
                   and now+blend_steps/self.fps-self.latest_captured > self.max_age):
                blend_steps -= 1
            require_fresh(self.latest_captured, self.max_age,
                          now=now+blend_steps/self.fps, label='Previous chunk through handoff')
            previous_age_at_handoff_end = now+blend_steps/self.fps-self.latest_captured
            # Blend time-aligned trajectories with a common C2 weight for all joints.
            # Finish within the overlap without shifting either chunk's time origin.
            for index in overlap:
                u = min(1., max(0., (index-step)/blend_steps))
                weight = u*u*u*(10.+u*(-15.+6.*u))
                future[index] = (1.-weight)*self.actions[index] + weight*future[index]
        self.measurements = dict(raw_step_peak_rad=peak, aligned_handoff_delta_rad=handoff,
                                 handoff_blend_steps=blend_steps,
                                 previous_prediction_age_at_handoff_end_s=previous_age_at_handoff_end)
        self.latest_origin = origin
        self.latest_captured = captured
        self.actions = future
        self.trajectory = {i: q for i, q in self.trajectory.items() if i >= step-math.ceil(self.max_age*self.fps)}
        self.trajectory.update(future)
        return True

    def rtc_prefix(self, origin, chunk_size, now):
        """Snapshot the admitted trajectory on the NEW observation's model-time grid."""
        if self.latest_origin is None:
            return None
        require_fresh(self.latest_captured, self.max_age, now=now, label='RTC previous prediction')
        prefix = []
        for i in range(chunk_size):
            tick = origin+i/self.speed_scale
            left = math.floor(tick+1e-9)
            fraction = max(0., tick-left)
            if left not in self.trajectory or (fraction > 1e-9 and left+1 not in self.trajectory):
                break
            a = self.trajectory[left]
            b = self.trajectory.get(left+1, a)
            prefix.append((1-fraction)*a+fraction*b)
        if not prefix:
            raise RuntimeError('RTC previous trajectory does not cover the observation time')
        result = np.array(prefix, dtype=np.float64)
        result.flags.writeable = False
        return result

    def pop(self, step):
        self.actions = {i: q for i, q in self.actions.items() if i >= step}
        return self.actions.pop(step, None)

    def pop_with_velocity(self, step):
        target = self.pop(step)
        if target is None:
            return None, None
        following = self.actions.get(step + 1, target)
        velocity = (following - target) * self.fps
        if self.locked_joints:
            target = target.copy()
            for joint, value in self.locked_joints.items():
                target[joint], velocity[joint] = value, 0.
        # Keep the tangent's joint ratios when one axis reaches its velocity bound.
        velocity /= max(1., float(np.max(np.abs(velocity)/self.velocity_limits)))
        # Division can round a saturated axis one ULP beyond the strict wire limit.
        velocity = np.clip(velocity, -self.velocity_limits, self.velocity_limits)
        return target, velocity


class AsyncPredictor:
    """Single model owner, one latest observation, one latest result; no backlog."""
    def __init__(self, predict, max_age=None, fresh_predict=None, on_event=None, initialize=None):
        self.predict = predict
        self.max_age, self.fresh_predict = max_age, fresh_predict
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.request = self.result = self.error = None
        self.request_id = 0
        self.busy = False
        self.next_observation = None
        self.last_origin = None
        self.on_event = on_event
        self.initialize = initialize
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def wait_ready(self, timeout=30.):
        if not self.ready.wait(timeout):
            raise RuntimeError('Inference worker initialization timed out')
        if self.error is not None:
            raise RuntimeError('Inference worker initialization failed') from self.error

    def can_submit(self):
        with self.lock:
            return not self.busy and self.request is None and self.result is None and self.error is None

    def submit(self, step, captured, observation, *, observation_sequence=None, state_stamp=None):
        with self.lock:
            replaced = self.request
            self.request_id += 1
            self.last_origin = step
            timing = dict(prediction_id=self.request_id, observation_sequence=observation_sequence,
                          captured_monotonic=captured, state_reference_monotonic=state_stamp,
                          submitted_monotonic=time.monotonic())
            self.request = (step, captured, observation, timing)
        if replaced is not None and self.on_event:
            self.on_event('inference_request_replaced', prediction_id=replaced[3]['prediction_id'],
                          replacement_prediction_id=timing['prediction_id'])
        self.wake.set()
        return timing['prediction_id']

    def take(self):
        with self.lock:
            if self.error:
                raise RuntimeError('ACT inference worker failed') from self.error
            result, self.result = self.result, None
            if result is not None:
                result[3]['result_consumed_monotonic'] = time.monotonic()
            return result

    def _run(self):
        try:
            if self.initialize is not None:
                self.initialize()
        except Exception as error:
            with self.lock:
                self.error = error
            return
        finally:
            self.ready.set()
        while not self.closed.is_set():
            self.wake.wait(0.1)
            self.wake.clear()
            with self.lock:
                request, self.request = self.request, None
                self.busy = request is not None
            if request is None:
                continue
            step, captured, obs, timing = request
            try:
                started = time.monotonic()
                timing['worker_started_monotonic'] = started
                timing['input_queue_s'] = started-timing['submitted_monotonic']
                timing['input_age_at_worker_start_s'] = started-captured
                checks = []
                def check():
                    checks.append(time.monotonic())
                    if self.max_age is not None:
                        require_fresh(captured, self.max_age, label='Inference input')
                check()
                actions = validate_targets(self.fresh_predict(obs, check) if self.fresh_predict else self.predict(*obs))
                finished = time.monotonic()
                timing.update(worker_finished_monotonic=finished, policy_execution_s=finished-started,
                              capture_to_prediction_ready_s=finished-captured,
                              before_last_input_check_s=checks[-1]-started,
                              after_last_input_check_s=finished-checks[-1], input_check_count=len(checks))
                if self.on_event:
                    self.on_event('inference_finished', **timing)
                supplier = self.next_observation
                with self.lock:
                    self.busy = supplier is not None
                    if not self.closed.is_set():
                        self.result = (step, captured, actions, timing)
                # Stateless policies can begin the successor before the control
                # thread consumes this result. RTC remains admission-driven.
                if supplier is not None and not self.closed.is_set():
                    successor = supplier()
                    with self.lock:
                        if (successor is not None and self.request is None and not self.closed.is_set()
                                and successor[0] > self.last_origin):
                            origin, stamp, observation, sequence, state_stamp = successor
                            self.request_id += 1
                            self.last_origin = origin
                            trace = dict(prediction_id=self.request_id, observation_sequence=sequence,
                                captured_monotonic=stamp, state_reference_monotonic=state_stamp,
                                submitted_monotonic=time.monotonic())
                            self.request = (origin, stamp, observation, trace)
                            self.wake.set()
                            if self.on_event:
                                self.on_event('inference_request_prefetched', origin=origin, **trace)
                        self.busy = self.request is not None
            except Exception as error:
                with self.lock:
                    self.busy = False
                    self.error = error
                return

    def close(self):
        self.closed.set()
        self.wake.set()
        self.next_observation = None
        with self.lock:
            self.request = self.result = None
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('Inference worker did not stop; process restart required')


def load_agent(project, checkpoint, device):
    project = Path(project).expanduser().resolve(strict=True)
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if (checkpoint / 'pretrained_model').is_dir():
        checkpoint /= 'pretrained_model'
    # Use the training project's exact antialiased Resize and saved native processors.
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location('fr3_hockey_inference', project / 'inference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.HockeyACT(checkpoint, device=device)
    metadata = agent.metadata
    if (metadata['fps'] != 30 or metadata['state_units'] != 'rad' or metadata['action_units'] != 'rad'
            or metadata['joint_names'] != [f'fr3_joint{i}' for i in range(1, 8)]
            or metadata['action_definition'] not in ('next_sample_joint_position_rad',
                                                     'next_uniform_time_interpolated_joint_position_rad')):
        raise ValueError('Checkpoint does not match the FR3 Teach 30 Hz absolute-radian contract')
    config = agent.policy.config
    if config.temporal_ensemble_coeff is not None:
        raise ValueError('This chunk runner requires temporal_ensemble_coeff=None')
    if set(config.input_features) not in (
            {'observation.state', 'observation.images.overhead'},
            {'observation.state', 'observation.images.wrist', 'observation.images.overhead'},
            {'observation.state', 'observation.images.wrist', 'observation.images.overhead', 'observation.images.third'}):
        raise ValueError('Expected state with overhead, wrist + overhead, or all three camera features')
    if tuple(config.input_features['observation.state'].shape) != (7,) or tuple(config.action_feature.shape) != (7,):
        raise ValueError('Expected seven state and action dimensions')
    agent.reset()
    return agent, checkpoint


class NativeLink:
    def __init__(self, port, execute, speed_scale=1.0):
        self.speed_scale = validate_speed_scale(speed_scale)
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect(('127.0.0.1', port))
        self.sock.setblocking(False)
        self.sequence = 0
        self.latest = None
        self.execute = execute
        self._initialize_receiver()
        self.sock.send(COMMAND.pack(HELLO, PROTOCOL_VERSION, *([0.] * 15)))

    def _initialize_receiver(self):
        self.receive_lock = threading.Lock()
        self.state_history = StateHistory()
        self.receiver_stop = threading.Event()
        self.receiver = self.receiver_error = None
        self.on_accept = None
        self.last_logged_acceptance = 0
        self.latest_timing = None

    def start_receiver(self, on_accept=None):
        self.on_accept = on_accept
        if self.receiver is not None:
            raise RuntimeError('Native receiver already started')
        def receive():
            try:
                while not self.receiver_stop.is_set():
                    self.poll()
                    self.receiver_stop.wait(.001)
            except BaseException as error:
                self.receiver_error = error
        self.receiver = threading.Thread(target=receive, daemon=True, name='fr3-native-state')
        self.receiver.start()

    def poll(self):
        if self.receiver_error is not None:
            raise RuntimeError(f'Native state receiver failed: {self.receiver_error}') from self.receiver_error
        with self.receive_lock:
            return self._poll()

    def _poll(self):
        for _ in range(64):
            try:
                data = self.sock.recv(2048)
            except BlockingIOError:
                break
            except ConnectionRefusedError as error:
                raise RuntimeError(
                    f'Native ACT bridge connection refused at 127.0.0.1:{self.port}. '
                    'Start/restart fr3_act_bridge in the ROS container and inspect its output; '
                    'it exits after a stop or a 60-second first-command timeout.'
                ) from error
            if data.startswith(b'FR3_ACT_ERROR: '):
                raise RuntimeError('Native ACT bridge stopped: ' + data[15:].decode('utf-8', errors='replace'))
            if len(data) != STATE.size:
                raise RuntimeError('Invalid native state packet')
            values = STATE.unpack(data)
            received = time.monotonic()
            if values[0] != STATUS or not all(math.isfinite(x) for x in values):
                raise RuntimeError('Invalid native state')
            if values[35] != PROTOCOL_VERSION or values[34] not in (0, 1, 2, 3, 4):
                raise RuntimeError('Native protocol/ready state mismatch; rebuild both ends')
            if values[3] != PROFILE_ID:
                raise RuntimeError('ACT limits differ from the native bridge; rebuild and restart the bridge')
            if not math.isclose(values[18], self.speed_scale, rel_tol=0, abs_tol=1e-9):
                raise RuntimeError(f'Speed scale differs: client={self.speed_scale:g}, bridge={values[18]:g}; '
                                   'restart both with the same --speed-scale')
            if values[1] != int(self.execute):
                raise RuntimeError('Real bridge requires --execute; simulation requires its omission')
            if values[19] > 0 and not (0 < values[36] <= values[37] <= values[38] <= values[40] <= received+.00001):
                raise RuntimeError('Invalid native command timing or mismatched monotonic clock')
            if self.on_accept is not None:
                self.on_accept("native_state", native_state_monotonic=values[2],
                    robot_time_s=values[39], measured_rad=list(values[4:11]),
                    sdk_desired_rad=list(values[11:18]), native_output_rad=list(values[27:34]),
                    accepted_sequence=int(values[19]), execute=bool(values[1]))
            self.state_history.append(values)
            self.latest = values
            self.latest_timing = (values[2], values[40], received)
            if self.on_accept is not None and values[19] > self.last_logged_acceptance:
                self.on_accept('native_acceptance', sequence=int(values[19]),
                    native_received_monotonic=values[36], native_accepted_monotonic=values[37],
                    native_output_monotonic=values[38], native_state_monotonic=values[2],
                    native_published_monotonic=values[40], ack_received_monotonic=received,
                    robot_time_s=values[39], accepted_target_rad=list(values[20:27]),
                    native_output_rad=list(values[27:34]), measured_rad=list(values[4:11]))
                self.last_logged_acceptance = values[19]
        if self.latest is None:
            return None
        age = time.monotonic() - self.latest[2]
        if age < -0.01:
            raise RuntimeError(f'Native state timestamp is {-age*1000:.1f} ms ahead; check host monotonic clocks')
        if age > 0.1:
            raise RuntimeError(f'Robot state is stale: age={age*1000:.1f} ms, limit=100 ms; '
                               'the native bridge may have aborted. Inspect its terminal output.')
        return np.array(self.latest[4:11])

    def send(self, target, velocity=None, *, captured=None, max_age=None, step=None, clock=None):
        target = validate_targets([target])[0]
        velocity = np.zeros(7) if velocity is None else np.asarray(velocity, dtype=np.float64)
        if velocity.shape != (7,) or not np.isfinite(velocity).all() or (np.abs(velocity) > VELOCITY_LIMITS*self.speed_scale).any():
            raise ValueError('Invalid target velocity')
        if captured is not None:
            require_fresh(captured, max_age, label='Active prediction')
        if self.latest is None:
            raise RuntimeError('No native ready state')
        require_fresh(self.latest[2], .1, label='Robot state before send')
        if self.latest[34] not in (0, 1):
            raise RuntimeError('Native bridge is stopping or read-only; restart and revalidate before rearm')
        if clock is not None:
            now = time.monotonic()
            if now < clock.due()-1e-7:
                raise RuntimeError('Control send interval compressed before dispatch')
            if clock.index(now) != step:
                raise ControlSlotExpired('Control slot expired before dispatch')
        self.sequence += 1
        deadline = time.monotonic() + COMMAND_WATCHDOG_SECONDS
        if captured is not None:
            deadline = min(deadline, captured + max_age)
        self.sock.send(COMMAND.pack(GOAL, self.sequence, deadline, *target, *velocity))
        return self.sequence

    def close(self):
        self.receiver_stop.set()
        if self.receiver is not None:
            self.receiver.join(timeout=1)
        try:
            for _ in range(3):
                self.sock.send(COMMAND.pack(STOP, *([0.] * 16)))
        except OSError:
            pass
        finally:
            self.sock.close()


def connect_native_bridge(port, execute, speed_scale, timeout=5.):
    """Retry refusal only before the first goal; never reconnect during motion."""
    print(f'Model and cameras ready. Waiting up to {timeout:g} s for native bridge at 127.0.0.1:{port}. '
          'Start the native bridge in the other terminal now.', flush=True)
    deadline = time.monotonic() + timeout
    link = None
    try:
        while time.monotonic() < deadline:
            if link is None:
                link = NativeLink(port, execute, speed_scale)
            try:
                q = link.poll()
            except RuntimeError as error:
                if not isinstance(error.__cause__, ConnectionRefusedError):
                    raise
                # No listener and no goals: do not send STOP to a newly started peer.
                link.sock.close()
                link = None
            else:
                if q is not None:
                    return link, q
            time.sleep(min(.05, max(0., deadline-time.monotonic())))
        raise RuntimeError(f'No native bridge state at 127.0.0.1:{port} within {timeout:g} seconds; '
                           'start the native bridge and inspect its terminal/log. No motion goals were sent.')
    except BaseException:
        if link is not None:
            link.close()
        raise


def agent_camera_names(agent):
    return tuple(name for name in ('wrist', 'overhead', 'third')
                 if f'observation.images.{name}' in agent.camera_keys)


def check_camera_mode(agent, mode):
    actual = agent_camera_names(agent)
    expected = {'overhead': ('overhead',), 'both': ('wrist', 'overhead'),
                'three': ('wrist', 'overhead', 'third')}.get(mode)
    if mode is not None and actual != expected:
        raise ValueError(f'Checkpoint cameras {actual} do not match requested mode {mode}')
    return actual


def validate_live_config(agent, config):
    if config['arm_joint_names'] != agent.metadata['joint_names']:
        raise ValueError('Live joint order differs from training')
    for name in agent_camera_names(agent):
        if name not in config['cameras']:
            raise ValueError(f'Missing live camera {name}; use the matching Teach camera configuration')
        camera = config['cameras'][name]
        if camera['fps'] != agent.metadata['fps']:
            raise ValueError(f'Live camera {name}.fps differs from training')
        source_size = agent.metadata.get('source_image_size')
        if agent.metadata['cameras'] == 'three' and source_size is not None:
            if [camera['height'], camera['width']] != source_size:
                raise ValueError(f'Live camera {name} must provide source RGB size {source_size}')


def camera_observation(cameras, config):
    from .observation_timing import capture_cameras
    return capture_cameras(cameras, config)


def offline_check(agent, checkpoint, max_start_distance=1.0):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    manifest = json.loads((checkpoint / 'hockey_dataset_manifest.json').read_text())
    root = Path(manifest['dataset_root'])
    dataset = LeRobotDataset(f'local/{root.name}', root=root, video_backend='pyav')
    from common import select_dataset_cameras
    select_dataset_cameras(dataset, agent.camera_keys)
    frame = dataset[0]
    q = frame['observation.state'].numpy()
    names = agent_camera_names(agent)
    arguments = ('wrist', 'overhead', 'third') if 'third' in names else ('wrist', 'overhead')
    images = [(frame[f'observation.images.{key}'].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
              if key in names else None for key in arguments]
    elapsed = []
    for _ in range(4):
        start = time.monotonic()
        chunk = validate_targets(agent.predict_chunk(q, *images))
        elapsed.append(time.monotonic() - start)
    print(json.dumps(dict(checkpoint=str(checkpoint), chunk_shape=list(chunk.shape),
                          first_inference_s=elapsed[0], subsequent_inference_s=elapsed[1:],
                          first_dataset_frame_start_checks=[target_distance_report(q, chunk[i], False, i, max_start_distance)
                                                            for i in range(min(3, len(chunk)))],
                          finite=True, contacts_robot=False), indent=2))


def run(options, agent, config):
    from franka_lerobot_recorder.realsense_camera import RealSenseColorCamera, resolve_realsense_serial
    cameras = {}
    link = worker = observer = journal = None
    observe_only = getattr(options, 'observe_only', False)
    max_start_distance = getattr(options, 'max_start_distance', 1.0)
    speed_scale = validate_speed_scale(getattr(options, 'speed_scale', 1.0))
    max_target_distance = validate_max_target_distance(getattr(options, 'max_target_distance', 1.0))
    profile = getattr(options, 'safety_profile_data', None)
    guard = RawActionGuard(None if profile is None else profile['max_raw_step_rad'],
                           None if profile is None else profile['max_handoff_rad'])
    diagnostic = None
    history = deque(maxlen=300)
    previous_target = None
    policy_name = getattr(options, 'policy_name', 'ACT')
    output = Path(getattr(options, 'diagnostic_output',
                          Path(__file__).resolve().parents[4] / 'artifacts/fr3-act-last-diagnostic.json'))
    status, failure = 'completed', None
    handoff_request = getattr(options, 'handoff_request', None)
    stage_status = getattr(options, 'stage_status', None)
    end_reason = 'duration'
    cleanup_errors = []
    old_handlers = {}
    ticks = commands = observed = skipped = 0
    fps = config['dataset']['fps']
    camera_age = config['timing']['max_camera_age_s']
    inference_mode = getattr(options, 'inference_mode', 'async')
    refresh_steps = getattr(agent, 'min_refresh_steps', 1)
    pipelined = inference_mode != 'rtc' and refresh_steps == 1
    try:
        if threading.current_thread() is threading.main_thread():
            def terminate(signum, _frame):
                raise KeyboardInterrupt(f'Signal {signum}')
            for signum in (signal.SIGTERM, signal.SIGHUP):
                old_handlers[signum] = signal.signal(signum, terminate)
        policy_config = getattr(getattr(agent, 'policy', None), 'config', None)
        journal = Journal(output.parent / (output.stem+'-runs'), dict(
            policy_name=policy_name, checkpoint=getattr(options, 'checkpoint_identity', None),
            motion_profile_id=PROFILE_ID, motion_limits=MOTION_LIMITS, safety_profile=profile,
            code_sha256={str(Path(__file__).name): file_digest(__file__),
                         'deployment_safety.py': file_digest(Path(__file__).with_name('deployment_safety.py')),
                         'observation_timing.py': file_digest(Path(__file__).with_name('observation_timing.py')),
                         'inference_cache.py': file_digest(Path(__file__).with_name('inference_cache.py'))},
            native_protocol=PROTOCOL_VERSION, capture_clock='host_monotonic',
            state_alignment='linear_interpolation_at_common_camera_reference',
            max_prediction_age_s=getattr(options, 'max_inference_age', 1.2), max_camera_input_age_s=camera_age,
            nominal_refresh_steps=getattr(policy_config, 'n_action_steps', None), physical_fps=fps,
            effective_refresh_steps=refresh_steps, refresh_policy='idle_worker_latest_observation',
            inference_prefetch=pipelined, execute=bool(getattr(options, "execute", False)),
            handoff_steps=getattr(options, "handoff_steps", 5),
            requested_joint_holds=[i+1 for i, flag in [(0,'lock_first_joint'),(4,'lock_fifth_joint')]
                                   if getattr(options,flag,False)],
            inference_cache=getattr(agent, 'metadata', {}).get('inference_cache'),
            speed_scale=speed_scale, max_target_distance=max_target_distance,
            running_target_distance_rad=(TARGET_DISTANCE if max_target_distance is None else np.full(7,max_target_distance)).tolist(), observe_only=observe_only,
            inference_mode=inference_mode, rtc_execution_horizon=getattr(options, 'rtc_execution_horizon', None),
            rtc_max_guidance_weight=getattr(options, 'rtc_max_guidance_weight', None),
            failure_policy='bounded stop if bridge/control available; otherwise SDK/hardware stop',
            rearm='restart both processes; fresh Idle/stationary checks; no automatic recovery'))
        camera_specs = [(key, resolve_realsense_serial(config['cameras'][key]['serial']))
                        for key in agent_camera_names(agent)]
        def start_camera(spec):
            key, (serial, label, model) = spec
            camera = config['cameras'][key]
            cameras[key] = RealSenseColorCamera(serial, camera['width'], camera['height'], camera['fps'],
                                                label=label, model_name=model)
        # Keep each camera's full exposure settling; overlap independent devices.
        # The pool joins all opens before cleanup can release partial successes.
        with ThreadPoolExecutor(max_workers=max(1, len(camera_specs))) as pool:
            for _ in pool.map(start_camera, camera_specs):
                pass
        camera_deadline = time.monotonic() + 5
        while True:
            try:
                frames, _ = camera_observation(cameras, config)
                break
            except RuntimeError:
                if time.monotonic() >= camera_deadline:
                    raise
                time.sleep(.01)
        warmup_state = np.array([0., 0., 0., -1.5, 0., 1.5, 0.])
        warmup_chunk = validate_targets(agent.predict_chunk(warmup_state, *frames))
        # Re-read cameras and clear only dynamic visual caches: repeated-image warmup
        # must not understate the cost of the next genuinely new observation.
        frames, warmup_anchor = camera_observation(cameras, config)
        if hasattr(agent, 'clear_visual_cache'):
            agent.clear_visual_cache()
        warmup_started = time.monotonic()
        warmup_age = warmup_started-getattr(frames, 'timing', {}).get('oldest_capture_monotonic', warmup_anchor)
        if inference_mode == 'rtc':
            validate_targets(agent.predict_chunk(warmup_state, *frames, rtc_context=dict(
                previous_actions=warmup_chunk, inference_delay=0,
                execution_horizon=options.rtc_execution_horizon)))
        else:
            validate_targets(agent.predict_chunk(warmup_state, *frames))
        warmup_latency = time.monotonic()-warmup_started
        budget = continuous_inference_budget(warmup_latency, warmup_age, options.max_inference_age, fps, refresh_steps, pipelined)
        journal.event('continuous_inference_budget', **budget)
        if not budget['fits'] and not observe_only:
            raise RuntimeError(f'Continuous inference budget does not fit: estimated {budget["required_age_s"]:.3f}s '
                               f'exceeds {options.max_inference_age:.3f}s; optimize inference before motion')
        if hasattr(agent, 'reset'):
            agent.reset()
        supports_freshness = 'freshness_check' in inspect.signature(agent.predict_chunk).parameters
        def predict_fresh(obs, check):
            if supports_freshness:
                return agent.predict_chunk(*obs, freshness_check=check)
            check()
            return agent.predict_chunk(*obs)
        if inference_mode == 'rtc':
            from franka_lerobot_recorder.pi05_deploy import RTCInference
            predict_fresh = RTCInference(agent, fps, speed_scale, options.max_inference_age,
                                          warmup_latency, journal)
        initialize_worker = (lambda: predict_fresh.warmup((warmup_state, *frames), options.rtc_execution_horizon, warmup_age)
                             ) if inference_mode == 'rtc' else None
        worker = AsyncPredictor(agent.predict_chunk, max_age=camera_age, fresh_predict=predict_fresh,
                                on_event=journal.event, initialize=initialize_worker)
        worker.wait_ready()
        if stage_status:
            atomic_json(stage_status, dict(state='waiting_bridge'))
        link, q = connect_native_bridge(options.port, options.execute, speed_scale,
                                        getattr(options, 'bridge_startup_timeout', 5.))
        locked_joints = {i:float(q[i]) for i, flag in [(0,'lock_first_joint'),(4,'lock_fifth_joint')]
                         if getattr(options,flag,False)}
        if locked_joints:
            journal.event('joint_hold_initialized', positions_rad={f'J{i+1}':v for i,v in locked_joints.items()},
                          source='initial_measured_joint_state')
            print('Joint position hold: '+', '.join(f'J{i+1}={v:.6f} rad' for i,v in locked_joints.items()),flush=True)
        link.start_receiver(journal.event)
        observation_sequence = 0
        def read_observation():
            nonlocal observation_sequence
            frames, anchor = camera_observation(cameras, config)
            q, alignment = link.state_history.interpolate(anchor,
                max_gap=config['timing'].get('max_state_interpolation_gap_s', .03))
            finished = time.monotonic()
            timing = dict(getattr(frames, 'timing', {}), **alignment)
            native_timing = getattr(link, 'latest_timing', None)
            if native_timing is not None:
                source, published, received = native_timing
                timing.update(state_transport_s=received-published, state_callback_to_publish_s=published-source,
                              state_age_at_observation_s=finished-source)
            captured = timing.get('oldest_capture_monotonic', anchor)
            require_fresh(anchor, camera_age, now=finished, label='Aligned observation state')
            require_fresh(captured, camera_age, now=finished)
            for frame in frames:
                if frame is not None:
                    frame.flags.writeable = False
            observation_sequence += 1
            timing.update(observation_finished_monotonic=finished, snapshot_age_s=finished-captured,
                          observation_build_s=finished-timing.get('camera_read_started_monotonic', finished))
            journal.event('observation', observation_sequence=observation_sequence, state_rad=q.tolist(), **timing)
            return Observation(observation_sequence, anchor, captured, finished, q, tuple(frames), timing)
        observer = ObservationPump(read_observation, fps)
        observer.latest(wait=True)  # Before any motion; running control uses latest() only.
        buffer = ActionBuffer(fps, options.max_inference_age, speed_scale, guard,
                              getattr(options, "handoff_steps", 5), locked_joints=locked_joints)
        started = False
        submitted = None
        submitted_origin = None
        rtc_pending = False
        clock = ControlClock(fps)
        if pipelined:
            def next_observation():
                snapshot = observer.refresh()
                origin = clock.index(snapshot.state_stamp)
                if origin <= worker.last_origin:
                    return None
                require_fresh(snapshot.captured, camera_age, label='Prefetched observation')
                return (origin, snapshot.captured, (snapshot.state, *snapshot.frames),
                        snapshot.sequence, snapshot.state_stamp)
            worker.next_observation = next_observation
        start = clock.epoch
        last_origin = last_captured = None
        active_timing = {}
        last_send_time = None
        while options.duration == 0 or time.monotonic() - start < options.duration:
            due = clock.due()
            time.sleep(max(0., due-time.monotonic()))
            now = time.monotonic()
            if started and now-due > COMMAND_WATCHDOG_SECONDS:
                raise RuntimeError('Control loop missed its deadline by more than 120 ms')
            step = clock.index(now)
            q = link.poll()
            journal.check()
            if handoff_request and started and Path(handoff_request).is_file():
                # A simultaneous operator request must not hide a policy failure.
                worker.take()
                end_reason = 'manual_handoff'
                journal.event('manual_handoff_requested', measured_rad=q.tolist())
                break
            snapshot = observer.latest()
            require_fresh(snapshot.captured, camera_age)
            if inference_mode == 'rtc' and not started:
                # Before the first goal, the prior trajectory is the stationary measured pose.
                buffer.trajectory[step] = q.copy()
            result = worker.take()
            if result is not None:
                origin, captured, actions, prediction_timing = result
                journal.event('prediction', origin=origin, captured=captured, actions=actions.tolist(),
                              inference_age_s=time.monotonic()-captured, **prediction_timing)
                # Check the full raw model horizon before selecting the execution prefix.
                guard.check_chunk(actions)
                if buffer.merge(origin, captured, actions[:options.actions_per_chunk], step, time.monotonic()):
                    last_origin, last_captured = origin, captured
                    active_timing = prediction_timing
                    journal.event('chunk_admitted', origin=origin, prediction_id=active_timing['prediction_id'],
                                  **buffer.measurements)
                current_budget = continuous_inference_budget(prediction_timing['policy_execution_s'],
                    prediction_timing['input_age_at_worker_start_s'], options.max_inference_age, fps, refresh_steps, pipelined)
                if not current_budget['fits']:
                    journal.event('continuous_inference_budget_overrun', **current_budget)
                    if not observe_only:
                        raise RuntimeError('Continuous inference budget overrun; stopping before prediction freshness is lost')
                rtc_pending = False
            target, velocity = buffer.pop_with_velocity(step)
            if target is not None:
                require_fresh(last_captured, options.max_inference_age, label='Active '+policy_name+' prediction')
                # Read feedback again after worker/result handling, immediately before admission.
                q = link.poll()
                report = target_distance_report(q, target, started, step, max_start_distance, max_target_distance)
                report.update(prediction_origin=last_origin, chunk_offset=(step-last_origin)*speed_scale,
                              prediction_id=active_timing.get('prediction_id'),
                              held_joints_rad={f'J{i+1}':v for i,v in locked_joints.items()},
                              speed_scale=speed_scale, prediction_age_s=time.monotonic()-last_captured,
                              sent_commands=commands, new_prediction=result is not None,
                              target_velocity_rad_s=velocity.tolist(),
                              target_step_delta_rad=None if previous_target is None else (target-previous_target).tolist(),
                              sdk_desired_rad=list(link.latest[11:18]), state_age_s=time.monotonic()-link.latest[2],
                              sdk_tracking_delta_rad=(q-np.array(link.latest[11:18])).tolist(),
                              accepted_sequence=int(link.latest[19]), accepted_target_rad=list(link.latest[20:27]),
                              native_output_rad=list(link.latest[27:34]), native_phase=int(link.latest[34]))
                if result is not None:
                    report['chunk_first_target_rad'] = actions[0].tolist()
                    report['chunk_max_step_delta_rad'] = guard.check_chunk(actions)
                history.append(report)
                previous_target = target.copy()
                journal.event('observation_and_target', **report)
                if observe_only:
                    observed += 1
                    if diagnostic is None:
                        diagnostic = dict(mode='observe_only', samples=0, first=report, worst=report)
                    diagnostic.update(samples=observed, last=report)
                    if report['max_abs_delta_rad'] > diagnostic['worst']['max_abs_delta_rad']:
                        diagnostic['worst'] = report
                else:
                    if report['exceeded_joints']:
                        diagnostic = dict(mode='target_rejected', **report)
                        raise TargetDistanceError(report, policy_name)
                    journal.event('send_intent', sequence=link.sequence+1, control_step=step,
                                  prediction_origin=last_origin, postprocess_target_rad=target.tolist())
                    journal.check()
                    require_fresh(snapshot.captured, camera_age, label='Snapshot before send')
                    send_start = time.monotonic()
                    try:
                        sequence = link.send(target, velocity, captured=last_captured,
                                             max_age=options.max_inference_age, step=step, clock=clock)
                    except ControlSlotExpired:
                        if started and time.monotonic()-due > COMMAND_WATCHDOG_SECONDS:
                            raise RuntimeError('Control loop missed its deadline by more than 120 ms')
                        # Discard this target, retaining the last real send time for spacing.
                        skipped += clock.skip(step)
                        journal.event('send_skipped', control_step=step, reason='slot_expired')
                        ticks += 1
                        continue
                    # Use send completion as spacing reference: conservatively bounds send START spacing.
                    send_end = time.monotonic()
                    skipped_now = clock.sent(send_end, step)
                    skipped += skipped_now
                    journal.event('send_dispatched', sequence=sequence, skipped_slots=skipped_now,
                                  prediction_id=active_timing['prediction_id'],
                                  captured_monotonic=last_captured,
                                  state_reference_monotonic=active_timing['state_reference_monotonic'],
                                  send_started_monotonic=send_start, send_finished_monotonic=send_end,
                                  capture_to_send_s=send_start-last_captured,
                                  inference_finish_to_send_s=send_start-active_timing['worker_finished_monotonic'],
                                  result_wait_s=active_timing['result_consumed_monotonic']-active_timing['worker_finished_monotonic'],
                                  send_duration_s=send_end-send_start,
                                  send_period_s=None if last_send_time is None else send_start-last_send_time)
                    last_send_time = send_start
                    if not started and stage_status:
                        atomic_json(stage_status, dict(state='hitting'))
                    started = True
                    commands += 1
            elif started:
                raise RuntimeError('Action queue exhausted; stopping instead of replaying stale targets')
            origin = clock.index(snapshot.state_stamp)
            if (worker.can_submit() and (inference_mode != 'rtc' or not rtc_pending)
                    and (submitted is None or step-submitted >= refresh_steps)
                    and (worker.last_origin is None or origin > worker.last_origin)):
                require_fresh(snapshot.captured, camera_age, label='Inference submission')
                # Actions are q(t+1): anchor to the measured state time, never to submission time.
                observation = (snapshot.state, *snapshot.frames)
                if inference_mode == 'rtc':
                    observation = (observation, dict(
                        previous_actions=buffer.rtc_prefix(origin, agent.policy.config.chunk_size, time.monotonic()),
                        previous_captured=buffer.latest_captured, state_stamp=snapshot.state_stamp,
                        origin_stamp=clock.epoch+origin/fps))
                prediction_id = worker.submit(origin, snapshot.captured, observation,
                    observation_sequence=snapshot.sequence, state_stamp=snapshot.state_stamp)
                rtc_pending = inference_mode == 'rtc'
                journal.event('inference_request', prediction_id=prediction_id, origin=origin, snapshot_sequence=snapshot.sequence,
                              state_stamp=snapshot.state_stamp, captured=snapshot.captured,
                              snapshot_finished=snapshot.finished, state_rad=snapshot.state.tolist(),
                              snapshot_age_s=time.monotonic()-snapshot.captured)
                submitted = step
                submitted_origin = origin
            ticks += 1
            if not started or observe_only:
                skipped += clock.sent(time.monotonic(), step)
        if not started and not (observe_only and observed):
            raise RuntimeError(f'No valid {policy_name} action became available')
        return dict(ticks=ticks, commands=commands, end_reason=end_reason)
    except BaseException as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error'
        failure = repr(error)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if worker is not None:
            worker.closed.set()
            worker.wake.set()
        # A cleanup failure must not prevent STOP, other cleanup, or evidence preservation.
        for resource in (link, observer, worker, *cameras.values()):
            if resource is not None:
                try:
                    resource.release() if resource in cameras.values() else resource.close()
                except BaseException as error:
                    cleanup_errors.append(repr(error))
        if history:
            if diagnostic is None:
                diagnostic = dict(mode='run', last=history[-1])
            diagnostic.update(policy_name=policy_name, history=list(history))
        if end_reason == 'manual_handoff' and worker is not None and worker.error is not None:
            cleanup_errors.append('Inference worker failed during handoff: '+repr(worker.error))
        diagnostic = dict(diagnostic or {}, status=status, error=failure, cleanup_errors=cleanup_errors,
                           end_reason=end_reason,
                          inference_mode=inference_mode,
                          ticks=ticks, commands=commands, skipped_slots=skipped,
                          run_directory=str(journal.path) if journal else None)
        try:
            atomic_json(output, diagnostic)
        except BaseException as error:
            cleanup_errors.append(repr(error))
        if journal is not None:
            try:
                journal.close(status, failure, ticks=ticks, commands=commands, skipped_slots=skipped,
                              cleanup_errors=cleanup_errors)
            except BaseException as error:
                cleanup_errors.append(repr(error))
        if cleanup_errors:
            print('Deployment cleanup/recording errors: '+ '; '.join(cleanup_errors), file=sys.stderr)
            if failure is None:
                raise RuntimeError('Deployment cleanup/recording failed: '+ '; '.join(cleanup_errors))
        if stage_status:
            atomic_json(stage_status, dict(state='stopped', status=status, end_reason=end_reason,
                                          cleanup_errors=cleanup_errors, error=failure))


def main(args=None, *, policy='act'):
    if policy not in ('act', 'pi05'):
        raise ValueError(f'Unsupported deployment policy: {policy}')
    parser = argparse.ArgumentParser(description=f'Asynchronous {policy} inference with the bounded FR3 executor')
    parser.add_argument('--project', default=f'/home/user1/projects/{policy}-fr3-hockey')
    if policy == 'pi05':
        parser.add_argument('--inference-mode', choices=('async', 'rtc'), default='async',
                            help='Async chunk generation, or async generation with RTC prefix guidance')
        parser.add_argument('--rtc-execution-horizon', type=int, default=10,
                            help='Model action steps over which RTC follows the previous trajectory')
        parser.add_argument('--rtc-max-guidance-weight', type=float, default=10.)
        parser.add_argument('--vision-precision', choices=('checkpoint','fp32','bf16'), default='checkpoint',
                            help='Frozen-expert visual compute; default follows the saved training precision')
        parser.add_argument('--attention-backend',choices=('sdpa','flash'),default=None,
                            help='Frozen-expert attention; flash skips masked padding keys, default sdpa')
        parser.add_argument('--task', help='Task text; defaults to the single task in the training dataset')
        parser.add_argument('--bridge-startup-timeout', type=float, default=5.,
                            help='Seconds to wait for native bridge after warmup; no goals while waiting')
        parser.add_argument('--dataset-root', type=Path, help='Relocated training dataset with recording contract')
        parser.add_argument('--allow-camera-replacement', action='append', default=[],
                            choices=('wrist', 'overhead', 'third'),
                            help='Explicitly allow a replaced camera serial; model, size, fps and orientation must match')
    parser.add_argument('--checkpoint', required=True, help='Explicit checkpoint; no moving last/best selection')
    parser.add_argument('--preflight-output', type=Path, help='Hash checkpoint and write contract without loading model/cameras/robot')
    parser.add_argument('--safety-profile', type=Path, help='Optional measured raw-step/handoff limits bound to checkpoint and motion profile; omitted: measure jumps without threshold enforcement')
    parser.add_argument('--cameras', choices=('overhead', 'both', 'three'), help='Verify checkpoint camera mode before connecting')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--config', required=True)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--duration', type=float, default=0.,
                        help='Control-loop seconds; 0 (default) runs continuously until interrupted')
    parser.add_argument('--actions-per-chunk', type=int, default=30)
    parser.add_argument('--handoff-steps', type=int, choices=range(1, 9), default=5)
    parser.add_argument('--max-inference-age', type=float, default=1.2,
                        help='Prediction age limit in seconds; default and maximum 1.2 for all policies')
    parser.add_argument('--lock-first-joint', action='store_true', help='Hold J1 at its initial measured position')
    parser.add_argument('--lock-fifth-joint', action='store_true', help='Hold J5 at its initial measured position')
    parser.add_argument('--max-target-distance', type=float, default=1.0,
                        help='Running per-joint target-distance limit (0, 1] rad; default: 1.0; match bridge')
    parser.add_argument('--max-start-distance', type=float, default=1.0,
                        help='Maximum first-target deviation per joint in radians, default 1.0; range (0, 1]')
    parser.add_argument('--speed-scale', type=float, default=1.0,
                        help='Slow action time and motion limits together, [0.1, 1]; must match native bridge')
    parser.add_argument('--offline-check', action='store_true', help='Dataset inference only; no cameras/robot/socket')
    parser.add_argument('--observe-only', action='store_true', help='Read live state/cameras and infer; never send motion goals')
    parser.add_argument('--diagnostic-output', type=Path,
                        default=Path(__file__).resolve().parents[4] / f'artifacts/fr3-{policy}-last-diagnostic.json')
    parser.add_argument('--execute', action='store_true', help='Permit connection to an explicitly started real bridge')
    options = parser.parse_args(args)
    options.policy_name = 'PI0.5' if policy == 'pi05' else 'ACT'
    if policy == 'pi05' and (options.rtc_execution_horizon < 1
                            or not math.isfinite(options.rtc_max_guidance_weight)
                            or options.rtc_max_guidance_weight <= 0):
        parser.error('RTC horizon and finite guidance weight must be positive')
    if policy == 'pi05' and (not math.isfinite(options.bridge_startup_timeout)
                            or not 0 < options.bridge_startup_timeout <= 300):
        parser.error('bridge-startup-timeout must be finite and within (0, 300] seconds')
    if not 1024 <= options.port <= 65535 or not math.isfinite(options.duration) or options.duration < 0:
        parser.error('Require valid port and finite nonnegative duration; 0 runs continuously')
    max_prediction_age = 1.2
    if not math.isfinite(options.max_inference_age) or not 0 < options.max_inference_age <= max_prediction_age:
        parser.error(f'max-inference-age must be in (0, {max_prediction_age:g}] seconds')
    if not math.isfinite(options.max_start_distance) or not 0 < options.max_start_distance <= 1:
        parser.error('max-start-distance must be finite and within (0, 1] rad')
    try:
        validate_speed_scale(options.speed_scale)
        validate_max_target_distance(options.max_target_distance)
    except ValueError as error:
        parser.error(str(error))
    from franka_lerobot_recorder.configuration import load_config
    config, _ = load_config(options.config)
    if config.get('collection_mode') != 'teach' or config['dataset']['fps'] != 30:
        parser.error('Use a 30 Hz FR3 Teach configuration')
    options.checkpoint_identity = checkpoint_identity(options.checkpoint, policy, getattr(options, 'dataset_root', None))
    identity = options.checkpoint_identity
    identity['inference_code_sha256'] = file_digest(Path(options.project)/'inference.py')
    if policy == 'pi05':
        identity.update(inference_mode=options.inference_mode,
                        rtc_execution_horizon=options.rtc_execution_horizon,
                        rtc_max_guidance_weight=options.rtc_max_guidance_weight,
                        rtc_runner_code_sha256=file_digest(Path(__file__).with_name('pi05_deploy.py')))
        if identity.get('checkpoint_format') == 'fr3_pi05_frozen_v1':
            if options.vision_precision != 'checkpoint':
                identity['vision_precision'] = options.vision_precision
            identity['attention_backend'] = options.attention_backend or 'sdpa'
            identity['sha256'] = hashlib.sha256(json.dumps(
                {'checkpoint': identity['sha256'], 'vision_precision': identity['vision_precision'],
                 'attention_backend':identity['attention_backend']},
                sort_keys=True).encode()).hexdigest()
        if options.inference_mode == 'rtc' and not 1 <= options.rtc_execution_horizon < options.actions_per_chunk:
            parser.error('RTC execution horizon must be smaller than actions-per-chunk')
        if options.inference_mode == 'rtc' and identity['n_action_steps'] >= options.actions_per_chunk:
            parser.error('RTC requires overlap: n_action_steps < actions-per-chunk')
    if not identity['n_action_steps'] <= options.actions_per_chunk <= identity['chunk_size']:
        parser.error('Require n_action_steps <= actions-per-chunk <= chunk_size')
    if options.preflight_output is not None:
        atomic_json(options.preflight_output, dict(identity, motion_profile_id=PROFILE_ID,
                                                   contacts_robot=False, loads_model=False))
        print(f'Checkpoint contract: {options.preflight_output}', flush=True)
        return
    if options.safety_profile is not None:
        options.safety_profile_data = load_safety_profile(options.safety_profile, identity['sha256'], PROFILE_ID)
    import torch
    torch.set_num_threads(2)
    if options.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this process; check GPU access or use --device cpu for offline checks')
    if policy == 'pi05':
        from franka_lerobot_recorder import pi05_deploy
        agent, checkpoint = pi05_deploy.load_agent(options.project, options.checkpoint, options.device,
                                                  dataset_root=options.dataset_root, task=options.task,
                                                  inference_mode=options.inference_mode,
                                                  rtc_execution_horizon=options.rtc_execution_horizon,
                                                  rtc_max_guidance_weight=options.rtc_max_guidance_weight,
                                                  vision_precision=options.vision_precision,
                                                  attention_backend=options.attention_backend)
        if not options.offline_check:
            pi05_deploy.validate_live_config(agent, config, options.allow_camera_replacement)
        print(f'Task: {agent.metadata["task"]}', flush=True)
        if agent.metadata.get('vision_precision'):
            print(f'Vision precision: {agent.metadata["vision_precision"]}; '
                  f'attention backend: {agent.metadata["attention_backend"]}',flush=True)
    else:
        agent, checkpoint = load_agent(options.project, options.checkpoint, options.device)
        validate_live_config(agent, config)
    try:
        camera_names = check_camera_mode(agent, options.cameras)
    except ValueError as error:
        parser.error(str(error))
    print(f'Checkpoint cameras: {camera_names}', flush=True)
    if not getattr(agent.policy.config, 'n_action_steps', 1) <= options.actions_per_chunk <= agent.policy.config.chunk_size:
        parser.error('Require n_action_steps <= actions-per-chunk <= checkpoint chunk_size')
    print(f'Checkpoint: {checkpoint}; device={options.device}; inference-mode='
          f'{getattr(options, "inference_mode", "async")}; no temporal ensembling', flush=True)
    print(f'Shared FR3 limits: {MOTION_LIMITS}; profile_id={PROFILE_ID}', flush=True)
    print(f'Startup target-distance limit: {options.max_start_distance:g} rad', flush=True)
    print(f'Speed scale: {options.speed_scale:g}; action time x{1/options.speed_scale:g}; '
          f'v x{options.speed_scale:g}, a x{options.speed_scale**2:g}, jerk x{options.speed_scale**3:g}; '
          'control/cameras remain 30 Hz', flush=True)
    if options.offline_check:
        if policy == 'pi05':
            pi05_deploy.offline_check(agent, checkpoint, options.max_start_distance, options.max_inference_age)
        else:
            offline_check(agent, checkpoint, options.max_start_distance)
    else:
        if options.observe_only:
            print('OBSERVE ONLY: no motion goals will be sent', flush=True)
        print('Running continuously; Ctrl+C requests braking' if options.duration == 0
              else f'Running for {options.duration:g} seconds', flush=True)
        run(options, agent, config)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; stop requested', file=sys.stderr)
        sys.exit(130)
