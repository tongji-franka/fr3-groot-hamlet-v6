"""Common monotonic timestamps, camera selection and bounded joint-state interpolation."""
from bisect import bisect_left
from collections import deque
import math
import threading
import time

import numpy as np


class ObservationNotReady(RuntimeError):
    """The first camera observation predates the available joint-state history."""


class CameraClock:
    """Convert SDK global_time to monotonic time using a bracketed host clock sample.

    SDK global_time is already mapped from the device to the OS wall clock:
    https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/h/rs_frame.h
    The reported bound covers the host clock read, not the SDK's device-clock fit.
    """
    def __init__(self):
        self.offset_ns = self.last_capture = self.last_frame = None
        self.failure = None

    def convert(self, device_ms, domain, frame_number, *, wall_ns=None, before_ns=None, after_ns=None):
        if before_ns is None:
            before_ns = time.monotonic_ns()
            wall_ns = time.time_ns()
            after_ns = time.monotonic_ns()
        if self.failure:
            raise RuntimeError(self.failure)
        span = after_ns - before_ns
        if not 0 <= span <= 2_000_000:
            raise RuntimeError('Host clock sample was interrupted for more than 2 ms')
        mono_ns = (before_ns + after_ns) // 2
        offset_ns = mono_ns - wall_ns
        if self.offset_ns is not None and abs(offset_ns - self.offset_ns) > 5_000_000:
            self.failure = 'Host wall clock stepped; restart capture after clock synchronization'
            raise RuntimeError(self.failure)
        self.offset_ns = offset_ns
        if str(domain).split('.')[-1] != 'global_time':
            raise RuntimeError(f'Camera requires SDK global_time for alignment, got {domain}')
        if not math.isfinite(device_ms) or device_ms < 0:
            raise RuntimeError('Invalid SDK frame timestamp')
        captured = (round(device_ms * 1_000_000) + offset_ns) / 1e9
        received = mono_ns / 1e9
        if captured > received or received - captured > 1.:
            raise RuntimeError('SDK capture time is in the future or more than one second old')
        if self.last_frame is not None and (frame_number <= self.last_frame or captured <= self.last_capture):
            raise RuntimeError('Camera frame number or capture timestamp did not advance')
        self.last_frame, self.last_capture = frame_number, captured
        return dict(received_wall=wall_ns / 1e9, received_monotonic=received,
                    device_timestamp_ms=device_ms, timestamp_domain=str(domain), device_frame_number=frame_number,
                    captured_monotonic=captured, capture_clock='monotonic',
                    capture_timestamp_source='sdk_global_time',
                    clock_error_bound_s=span / 2e9 + .0000005,
                    capture_to_receive_s=received-captured)


class CameraFrames(list):
    """Preserve the existing (frames, anchor) API, with per-camera timing attached."""
    def __init__(self, frames, metadata, anchor, started, finished):
        super().__init__(frames)
        self.metadata = metadata
        captures = [m['captured_monotonic'] for m in metadata.values()]
        self.timing = dict(camera_reference_monotonic=anchor, oldest_capture_monotonic=min(captures),
                           camera_capture_skew_s=max(captures)-min(captures),
                           camera_alignment_error_s=max(abs(t-anchor) for t in captures),
                           camera_read_started_monotonic=started, camera_read_finished_monotonic=finished,
                           camera_read_s=finished-started, cameras=metadata)


def capture_cameras(cameras, config):
    if not cameras:
        raise ValueError('At least one camera is required for a camera observation')
    started = time.monotonic()
    histories = {}
    copied = {}
    for name, camera in cameras.items():
        # Old camera interfaces can supply one timestamped frame. RealSense supplies
        # immutable references from a short history; only the selected frame is copied.
        history_api = getattr(type(camera), 'read_history', None)
        if history_api is not None:
            samples = camera.read_history()
            copied[name] = False
        else:
            sample = camera.read_sample(rotate_180=config['cameras'][name]['rotate_180'])
            samples = [] if sample is None else [sample]
            copied[name] = True
        if not samples:
            raise RuntimeError(f'No {name} camera frame')
        for _, metadata in samples:
            stamp = metadata.get('captured_monotonic')
            if (metadata.get('poll_error') or metadata.get('capture_clock') != 'monotonic'
                    or stamp is None or not math.isfinite(stamp)):
                raise RuntimeError(f'Invalid/failed {name} camera capture clock: {metadata.get("poll_error")}')
        histories[name] = samples
    now = time.monotonic()
    limit = config['timing']['max_camera_age_s']
    for name, samples in histories.items():
        if not 0 <= now-samples[-1][1]['captured_monotonic'] <= limit:
            raise RuntimeError(f'Stale {name} camera capture')
    # Each camera can bracket this reference with frames that have already arrived.
    anchor = min(samples[-1][1]['captured_monotonic'] for samples in histories.values())
    fps = config.get('dataset', {}).get('fps', 30)
    max_error = config['timing'].get('max_camera_alignment_error_s', .5/fps + .001)
    if not math.isfinite(max_error) or max_error <= 0:
        raise ValueError('Camera alignment tolerance must be positive and finite')
    frames, metadata = {}, {}
    for name, samples in histories.items():
        frame, stamp = min(samples, key=lambda item: abs(item[1]['captured_monotonic']-anchor))
        if abs(stamp['captured_monotonic']-anchor) > max_error:
            raise RuntimeError(f'{name} camera history cannot align to the common capture time')
        if not copied[name]:
            if config['cameras'][name]['rotate_180']:
                frame = np.ascontiguousarray(frame[::-1, ::-1])
            elif frame.flags.writeable or not frame.flags.owndata:
                frame = frame.copy()
            # RealSense already owns immutable pixels; retaining that reference
            # avoids another copy and preserves identity for repeated SDK frames.
        frames[name] = frame
        metadata[name] = dict(stamp, alignment_offset_s=stamp['captured_monotonic']-anchor)
    finished = time.monotonic()
    stamps = [m['captured_monotonic'] for m in metadata.values()]
    if not 0 <= finished-min(stamps) <= limit:
        raise RuntimeError('Stale camera snapshot after copying frames')
    if max(stamps)-min(stamps) > config['timing']['max_camera_skew_s']:
        raise RuntimeError('Camera capture skew exceeds configured limit')
    names = ('wrist', 'overhead', 'third') if 'third' in frames else ('wrist', 'overhead')
    return CameraFrames([frames.get(name) for name in names], metadata, anchor, started, finished), anchor


class StateHistory:
    """Retain every received native sample; never extrapolate a joint observation."""
    def __init__(self, capacity=512):
        self.samples = deque(maxlen=capacity)
        self.condition = threading.Condition()

    def append(self, packet):
        stamp = packet[2]
        q = np.array(packet[4:11], dtype=np.float64)
        if not math.isfinite(stamp) or q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError('Invalid native state history sample')
        q.flags.writeable = False
        with self.condition:
            if self.samples:
                last = self.samples[-1]
                if stamp < last[0] or (stamp == last[0] and not np.array_equal(q, last[1])):
                    raise RuntimeError('Native state timestamps regressed or changed without advancing')
                if stamp == last[0]:
                    return
            robot_time = packet[39] if len(packet) > 39 else None
            self.samples.append((stamp, q, robot_time))
            self.condition.notify_all()

    def interpolate(self, stamp, max_gap=.03, timeout=.05):
        if (not math.isfinite(stamp) or not math.isfinite(max_gap) or max_gap <= 0
                or not math.isfinite(timeout) or timeout < 0):
            raise ValueError('Invalid observation reference time or interpolation limits')
        deadline = time.monotonic()+timeout
        with self.condition:
            while not self.samples or self.samples[-1][0] < stamp:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('State history has no sample after the camera reference; refusing extrapolation')
                self.condition.wait(remaining)
            if stamp < self.samples[0][0]:
                raise ObservationNotReady('Camera reference predates joint-state history')
            samples = list(self.samples)
        right_index = bisect_left([item[0] for item in samples], stamp)
        right = samples[right_index]
        left = right if right[0] == stamp else samples[right_index-1]
        gap = right[0]-left[0]
        if gap > max_gap:
            raise RuntimeError(f'Joint-state interpolation gap {gap:.6f}s exceeds {max_gap:.6f}s')
        fraction = (stamp-left[0])/gap if gap else 0.
        q = (1-fraction)*left[1]+fraction*right[1]
        q.flags.writeable = False
        return q, dict(state_reference_monotonic=stamp, state_left_monotonic=left[0],
                       state_right_monotonic=right[0], state_interpolation_gap_s=gap,
                       state_interpolation_weight=fraction, state_left_robot_time_s=left[2],
                       state_right_robot_time_s=right[2],
                       state_time_basis='native_callback_receipt_monotonic')
