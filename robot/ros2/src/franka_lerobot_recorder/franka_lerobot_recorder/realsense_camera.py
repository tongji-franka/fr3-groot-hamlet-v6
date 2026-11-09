#!/usr/bin/env python3
"""RealSense 彩色相机的统一采集入口（librealsense 通路）。

本项目所有录制脚本共用这一份实现，相机一律按 **SDK 序列号** 定位
（``rs-enumerate-devices`` 显示的那个）。相比直接开 V4L2 节点，librealsense：

  - 用 ``enable_device(serial)`` + ``enable_stream(rs.stream.color, ...)`` 直接
    点名彩色流，不需要猜哪个 ``/dev/videoN`` 是彩色。同一台 RealSense 会同时
    暴露深度(Z16)/红外(GREY,UYVY)/彩色(YUYV) 多个节点，猜错就把深度图录进数据集。
  - 显式复位 brightness/contrast/gamma 等并重开自动曝光与自动白平衡，不继承
    上一个程序留下的极端参数（直开 V4L2 会过曝，而 Viewer 里看着正常）。
  - 全程不碰 ``/dev``，天然免疫 USB 插拔导致的 videoN 重新编号。

为兼容历史命令，``resolve_realsense_serial`` 也接受旧的 V4L2/udev 序列号，
会通过两者共同的 USB 物理端口映射到同一台 SDK 设备。
"""

import glob
from collections import deque
import os
import re
import subprocess
import threading
import time
from typing import Optional

import numpy as np
import pyrealsense2 as rs

from .observation_timing import CameraClock


def exposure_metadata(frame):
    """Keep raw device metadata; these fields are not host/global timestamps."""
    result = {}
    for sdk_name, field in [('sensor_timestamp', 'sensor_timestamp_us'),
                            ('frame_timestamp', 'frame_timestamp_us'),
                            ('actual_exposure', 'actual_exposure_us'),
                            ('actual_fps', 'actual_fps_x1000')]:
        key = getattr(rs.frame_metadata_value, sdk_name, None)
        try:
            if key is not None and frame.supports_frame_metadata(key):
                value = float(frame.get_frame_metadata(key))
                if np.isfinite(value) and value >= 0:
                    result[field] = value
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
            continue
    return result


def _v4l2_serial(dev: str) -> Optional[str]:
    """读取 v4l2 节点所属 USB 设备的序列号。拿不到返回 None。

    部分 RealSense/udev 组合不会在 ``--query=property`` 里生成
    ID_SERIAL_SHORT，也不会创建 /dev/v4l/by-id；此时继续沿 sysfs 父设备向上找
    ATTRS{serial}。D405/D435i 的硬件序列号位于 USB 父设备上。
    """
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "--query=property", "--name", dev],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("ID_SERIAL_SHORT="):
            return line.split("=", 1)[1].strip()

    try:
        out = subprocess.check_output(
            ["udevadm", "info", "--attribute-walk", "--name", dev],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('ATTRS{serial}=='):
            return line.split("==", 1)[1].strip().strip('"')
    return None


def _v4l2_has_color(dev: str) -> bool:
    """该节点能否输出彩色流(YUYV/MJPG)，用于排除深度/红外/metadata 节点。

    没有 v4l2-ctl 时不做过滤（返回 True），交给后续真正 open 时验证。
    """
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-formats", "-d", dev],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return True
    return ("YUYV" in out) or ("MJPG" in out)


def resolve_v4l2_device(spec):
    """把相机标识解析成可被 cv2.VideoCapture 打开的设备。

    /dev/videoN 编号每次重启/插拔都可能变（内核枚举顺序不固定），用整数 index
    指定相机会导致左右/顶画面错位。改用相机自身永不变的 USB 序列号定位。

    只接受 USB 序列号(如 "260323072579")：扫描所有 /dev/video* 找序列号匹配且
    能出彩色流的节点，编号怎么变都对得上。/dev/videoN 路径与整数 index 一律
    拒绝——历史上就是靠 index 指定相机，把 left/top 画面录错位过。

    返回 (device, desc)：device 传给 VideoCapture，desc 仅用于日志。
    """
    spec = str(spec).strip()

    if spec.startswith("/dev/"):
        raise RuntimeError(
            f"相机不再接受设备路径 {spec!r}；请用 USB 序列号指定。\n"
            "  /dev/videoN 编号每次重启/插拔都会变，历史上因此录错过机位。\n"
            "  查序列号：rs-enumerate-devices -s"
        )
    if spec.lstrip("-").isdigit() and len(spec.lstrip("-")) <= 5:
        raise RuntimeError(
            f"相机不再接受整数 index {spec!r}；请用 USB 序列号指定。\n"
            "  查序列号：rs-enumerate-devices -s"
        )

    nodes = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int("".join(filter(str.isdigit, p)) or 0),
    )
    serial_by_node = {dev: _v4l2_serial(dev) for dev in nodes}

    # 候选节点 = udev 序列号匹配的 ∪ /dev/v4l/by-id 链接指向的。两边都要查：
    # RealSense 有的节点读不到 ID_SERIAL_SHORT，而彩色节点又可能没有 by-id 链接
    # （本机 D435I 的 YUYV 节点 /dev/video4 就没有）。只查一边都会漏。
    candidates = [dev for dev, serial in serial_by_node.items() if serial == spec]
    for link in sorted(glob.glob(f"/dev/v4l/by-id/*{spec}*")):
        target = os.path.realpath(link)
        if target.startswith("/dev/video") and target not in candidates:
            candidates.append(target)

    # 必须挑能出彩色流的节点。一台 RealSense 会同时暴露深度(Z16)、红外(GREY/UYVY)
    # 和彩色(YUYV/MJPG) 若干节点，选错就把深度图当画面录进数据集 —— 之前按 by-id
    # 顺序取第一个，D435I 拿到的正是深度节点 /dev/video0。
    color = [dev for dev in candidates if _v4l2_has_color(dev)]
    if color:
        return color[0], f"serial {spec} -> {color[0]}"
    if candidates:
        raise RuntimeError(
            f"序列号 {spec!r} 对应节点 {candidates}，但没有一个能输出彩色流"
            "(YUYV/MJPG)。\n"
            "  不要把深度/红外节点当画面录制；请用 "
            "v4l2-ctl --list-formats -d <节点> 逐个确认。"
        )

    detected = ", ".join(
        f"{dev}={serial or '无序列号'}" for dev, serial in serial_by_node.items()
    ) or "未发现 /dev/video*"
    raise RuntimeError(
        f"找不到序列号为 {spec!r} 的相机。\n"
        f"  当前 V4L 节点：{detected}\n"
        "  该相机可能没插上；查序列号：rs-enumerate-devices -s"
    )


def _usb_port_key(path: str) -> Optional[str]:
    """从 sysfs 路径提取稳定的 USB 控制器/端口，例如 ``usb4/4-1``。"""
    match = re.search(r"/(usb\d+)/(\d+-[\d.]+)/", str(path))
    return f"{match.group(1)}/{match.group(2)}" if match else None


def _retry_usb_control(operation, label):
    """Retry transient UVC errors at startup; persistent errors remain fatal."""
    for attempt in range(3):
        try:
            return operation()
        except RuntimeError as exc:
            transient = any(message in str(exc).lower() for message in (
                'timed out', 'protocol error', 'errno=110', 'errno=71'))
            if not transient or attempt == 2:
                raise RuntimeError(f'RealSense [{label}]: {exc}') from exc
            print(f'[Camera {label}] USB control error: {exc}; retry {attempt + 1}/2', flush=True)
            time.sleep(.25 * (attempt + 1))


def _set_sensor_option(sensor, option, value, label):
    def configure():
        # UVC writes can time out even when an auto control is already enabled.
        if sensor.get_option(option) != value:
            sensor.set_option(option, value)
    _retry_usb_control(configure, f'{label} {option}')


def resolve_realsense_serial(spec: str) -> tuple[str, str, str]:
    """把 SDK 序列号或旧 V4L2 序列号解析为 librealsense 序列号。

    这台机器上的 D405/D435i 在 udev 与 librealsense 中会显示不同的序列号。
    为保持既有命令行兼容，先尝试直接匹配 SDK 序列号；否则沿用旧的 V4L2
    序列号解析，再通过二者共同的 USB 物理端口找到同一台 SDK 设备。
    """
    requested = str(spec).strip()
    devices = _retry_usb_control(lambda: list(rs.context().query_devices()),
                                 f'enumerating devices for serial {requested}')
    sdk_devices = []
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        physical_port = (
            device.get_info(rs.camera_info.physical_port)
            if device.supports(rs.camera_info.physical_port)
            else ""
        )
        sdk_devices.append((serial, name, physical_port))
        if serial == requested:
            return serial, f"{name}, SDK serial {serial}", name

    # 兼容脚本原有的 udev/V4L2 序列号（例如 260323073119）。
    v4l_device, _ = resolve_v4l2_device(requested)
    if isinstance(v4l_device, int):
        v4l_device = f"/dev/video{v4l_device}"
    try:
        sysfs_path = subprocess.check_output(
            ["udevadm", "info", "--query=path", "--name", str(v4l_device)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        raise RuntimeError(
            f"无法把 V4L2 序列号 {requested!r} 映射到 librealsense 设备：{exc}"
        ) from exc

    requested_port = _usb_port_key(sysfs_path)
    matches = [
        (serial, name)
        for serial, name, physical_port in sdk_devices
        if requested_port is not None and _usb_port_key(physical_port) == requested_port
    ]
    if len(matches) == 1:
        serial, name = matches[0]
        return (
            serial,
            f"{name}, V4L2 serial {requested} -> SDK serial {serial} "
            f"({requested_port})",
            name,
        )

    detected = ", ".join(
        f"{name}: SDK serial={serial}, port={_usb_port_key(port) or port or 'unknown'}"
        for serial, name, port in sdk_devices
    ) or "未发现 RealSense 设备"
    raise RuntimeError(
        f"无法唯一映射相机 {requested!r}（V4L2={v4l_device}, "
        f"port={requested_port or 'unknown'}）。当前 SDK 设备：{detected}"
    )


class RealSenseColorCamera:
    """通过 librealsense 采集 RGB，并显式恢复 Viewer 风格的自动控制。

    之前复用的 OpenCVCamera 直接打开 V4L2 节点，只配置分辨率、帧率和
    YUYV，没有初始化曝光、增益或白平衡。RealSense 相机会保留上一程序留下
    的控制状态，因此 Viewer 正常而录制画面可能严重过曝。这里改用与
    realsense-viewer 相同的 librealsense，并在开始采帧前重置颜色参数、打开
    自动曝光和自动白平衡。
    """

    _DEFAULT_COLOR_OPTIONS = (
        rs.option.backlight_compensation,
        rs.option.brightness,
        rs.option.contrast,
        rs.option.hue,
        rs.option.saturation,
        rs.option.sharpness,
        rs.option.gamma,
        rs.option.power_line_frequency,
    )

    def __init__(
        self,
        serial: str,
        width: int,
        height: int,
        fps: int,
        label: str = None,
        warmup_seconds: float = 3.0,
        model_name: str = "",
    ):
        self.serial = str(serial).strip()
        self.label = label or f"serial {self.serial}"
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        # RGB8 由 librealsense 完成 D405/D435i 的颜色转换，与 Viewer 使用同一
        # 条采集链路；read_rgb() 因此不再需要 OpenCV 的 BGR/RGB 转换。
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        # D405 的彩色流来自 Stereo Module，而非 D435i 那样的独立 RGB Camera。
        # 本机实测：D405 仅开 Color 时，即使 AE=on 仍有 52% 以上像素过曝；
        # 同时启用同 profile 的 Depth 或双 IR 后高光截断降为 0%。Viewer/ROS 也
        # 会为 D405 启动 Stereo Module 配套流。这里选单路 Depth，USB 带宽低于
        # 双 IR；Depth 只用于让 D405 正确出彩色，不会写入 LeRobot 数据集。
        self._uses_d405_companion = "D405" in model_name.upper()
        if self._uses_d405_companion:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        try:
            profile = self.pipeline.start(config)
            self._configure_viewer_defaults(profile.get_device())

            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            actual_width, actual_height = stream.width(), stream.height()
            actual_fps = stream.fps()
            intrinsics = stream.get_intrinsics()
            self.calibration = {
                'serial': self.serial, 'model': model_name,
                'width': actual_width, 'height': actual_height, 'fps': actual_fps,
                'fx': intrinsics.fx, 'fy': intrinsics.fy,
                'ppx': intrinsics.ppx, 'ppy': intrinsics.ppy,
                'distortion_model': str(intrinsics.model), 'coeffs': list(intrinsics.coeffs),
            }
            print(
                f"[Camera {self.label}] librealsense RGB8 "
                f"{actual_width}x{actual_height}@{actual_fps}"
                + (" + D405 Depth companion" if self._uses_d405_companion else "")
            )

            # 自动曝光刚打开时前几帧仍可能沿用旧曝光。按时间而不是固定 10 帧
            # 预热，保证 15/30 fps 下都至少有数秒用于收敛。
            warmup_frames = max(1, int(round(max(0.0, warmup_seconds) * actual_fps)))
            print(
                f"[Camera {self.label}] 等待自动曝光/白平衡收敛 "
                f"({warmup_frames} frames)..."
            )
            for _ in range(warmup_frames):
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if not frames.get_color_frame():
                    raise RuntimeError("librealsense 未返回彩色帧")
        except BaseException:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            raise

        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_metadata = None
        self._history = deque(maxlen=6)
        self._capture_clock = CameraClock()
        self._first_monotonic = None
        self._last_error = None
        # 每帧到手时刻的墙钟时间戳与累计帧数，供 check_camera_sync.py 量多路
        # 相机的时间对齐；录制主循环不用，开销只是一次 time.time()。
        self._latest_ts: Optional[float] = None
        self._first_ts: Optional[float] = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _configure_viewer_defaults(self, device) -> None:
        auto_exposure_configured = False
        settings = []

        color_sensors = []
        for sensor in device.query_sensors():
            try:
                if any(
                    profile.stream_type() == rs.stream.color
                    for profile in sensor.get_stream_profiles()
                ):
                    color_sensors.append(sensor)
            except Exception:
                continue

        if not color_sensors:
            raise RuntimeError(f"RealSense [{self.label}] 找不到彩色传感器")

        for sensor in color_sensors:
            try:
                sensor_name = sensor.get_info(rs.camera_info.name)
            except Exception:
                sensor_name = "sensor"

            if sensor.supports(rs.option.global_time_enabled):
                _set_sensor_option(sensor, rs.option.global_time_enabled, 1., self.label)
                settings.append(f"{sensor_name}.global_time=on")

            # Viewer 的颜色调节默认值来自 librealsense option range。显式恢复
            # 这些值，避免继承其他程序设置的极端 brightness/gamma 等参数。
            for option in self._DEFAULT_COLOR_OPTIONS:
                if not sensor.supports(option):
                    continue
                try:
                    option_range = sensor.get_option_range(option)
                    _set_sensor_option(sensor, option, option_range.default, self.label)
                    settings.append(f"{sensor_name}.{option}={option_range.default:g}")
                except RuntimeError as exc:
                    # 个别固件会声明某项受支持但在流启动后变为只读。颜色附加项
                    # 失败不应掩盖核心自动曝光；日志中明确记录即可。
                    settings.append(f"{sensor_name}.{option}=unchanged({exc})")

            if sensor.supports(rs.option.enable_auto_exposure):
                _set_sensor_option(sensor, rs.option.enable_auto_exposure, 1., self.label)
                auto_exposure_configured = True
                settings.append(f"{sensor_name}.auto_exposure=on")

            # 关闭 auto-exposure priority，避免曝光不足时通过降低实际帧率补偿；
            # RealSense Viewer 的默认值也是关闭。
            if sensor.supports(rs.option.auto_exposure_priority):
                _set_sensor_option(sensor, rs.option.auto_exposure_priority, 0., self.label)
                settings.append(f"{sensor_name}.auto_exposure_priority=off")

            if sensor.supports(rs.option.enable_auto_white_balance):
                _set_sensor_option(sensor, rs.option.enable_auto_white_balance, 1., self.label)
                settings.append(f"{sensor_name}.auto_white_balance=on")

        if not auto_exposure_configured:
            raise RuntimeError(
                f"RealSense [{self.label}] 没有可用的自动曝光控制，拒绝在曝光状态未知时录制"
            )
        print(f"[Camera {self.label}] " + ", ".join(settings))

    def _grab_loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                metadata = self._capture_clock.convert(color.get_timestamp(), color.get_frame_timestamp_domain(),
                                                       color.get_frame_number())
                metadata['device_exposure_metadata'] = exposure_metadata(color)
                rgb = np.asanyarray(color.get_data()).copy()
                metadata['copied_monotonic'] = time.monotonic()
                ts = metadata['received_wall']
                if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
                    # config 请求的是精确分辨率，出现不一致说明设备/SDK 配置异常；
                    # 不静默 resize，以免数据集声明与真实相机流不一致。
                    continue
                with self._lock:
                    rgb.flags.writeable = False
                    self._latest_rgb = rgb
                    self._latest_ts = ts
                    self._latest_metadata = metadata
                    self._last_error = None
                    self._history.append((rgb, metadata))
                    self._frame_id += 1
                    if self._first_ts is None:
                        self._first_ts = ts
                        self._first_monotonic = metadata['received_monotonic']
            except RuntimeError as error:
                if self._running:
                    with self._lock:
                        self._last_error = str(error)
                    continue

    def read_sample(self, rotate_180=False):
        """Atomically return pixels and timing for the same frame.

        Capture and receive times are distinct, both mapped to host monotonic time.
        """
        with self._lock:
            if self._latest_rgb is None:
                return None
            rgb = self._latest_rgb.copy()
            metadata = dict(self._latest_metadata)
            metadata['poll_error'] = self._last_error
        if rotate_180:
            rgb = np.ascontiguousarray(rgb[::-1, ::-1])
        return rgb, metadata

    def read_history(self):
        """Return immutable owned pixels; consumers copy only the selected frame."""
        with self._lock:
            return [(rgb, dict(metadata, poll_error=self._last_error)) for rgb, metadata in self._history]

    def read_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_rgb is None:
                return None
            return self._latest_rgb.copy()

    def read_stamp(self) -> Optional[tuple[float, int]]:
        """返回 (最新帧的墙钟时间戳, 累计帧序号)；还没抓到首帧时返回 None。"""
        with self._lock:
            if self._latest_ts is None:
                return None
            return self._latest_ts, self._frame_id

    def measured_fps(self) -> float:
        """实测抓帧速率，用于判断某一路是否掉速。"""
        with self._lock:
            if self._first_ts is None or self._frame_id < 2:
                return 0.0
            span = time.monotonic() - self._first_monotonic
            return (self._frame_id - 1) / span if span > 0 else 0.0

    def release(self):
        self._running = False
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
