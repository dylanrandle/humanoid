"""ROS 2 hardware-edge adapter for Meta Quest controller input."""

from __future__ import annotations

import importlib
import math
import subprocess
import time
from collections.abc import Callable
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Joy

from triskel_operator.quest_reader import QuestReaderSession
from triskel_operator.topics import QUEST_JOY_TOPIC, QUEST_POSE_TOPIC

ReaderFactory = Callable[[str, int], QuestReaderSession]
MAX_ADB_PORT = 65535
READER_CLOSE_TIMEOUT_SECONDS = 0.25


class MetaQuestBridge(Node):
    """Publish Meta Quest reader samples as standard ROS pose and joystick messages."""

    def __init__(self, reader_factory: ReaderFactory) -> None:
        super().__init__("meta_quest_bridge")
        self.declare_parameter("poll_rate_hz", 60.0)
        self.declare_parameter("reconnect_period", 2.0)
        self.declare_parameter("sample_timeout", 1.0)
        self.declare_parameter("frame_id", "openxr")
        self.declare_parameter("ip_address", "auto")
        self.declare_parameter("adb_port", 5555)
        self.declare_parameter("pose_topic", QUEST_POSE_TOPIC)
        self.declare_parameter("joy_topic", QUEST_JOY_TOPIC)

        poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        if not math.isfinite(poll_rate_hz) or poll_rate_hz <= 0.0:
            raise ValueError("poll_rate_hz must be positive and finite")
        self._reconnect_period = float(self.get_parameter("reconnect_period").value)
        self._sample_timeout = float(self.get_parameter("sample_timeout").value)
        if not math.isfinite(self._reconnect_period) or self._reconnect_period <= 0.0:
            raise ValueError("reconnect_period must be positive and finite")
        if not math.isfinite(self._sample_timeout) or self._sample_timeout <= 0.0:
            raise ValueError("sample_timeout must be positive and finite")
        self._frame_id = str(self.get_parameter("frame_id").value)
        configured_ip = str(self.get_parameter("ip_address").value).strip()
        self._ip_address = "" if configured_ip == "auto" else configured_ip
        self._adb_port = int(self.get_parameter("adb_port").value)
        if not 1 <= self._adb_port <= MAX_ADB_PORT:
            raise ValueError(f"adb_port must be between 1 and {MAX_ADB_PORT}")
        pose_topic = str(self.get_parameter("pose_topic").value)
        joy_topic = str(self.get_parameter("joy_topic").value)
        self._reader_factory = reader_factory
        self._reader: QuestReaderSession | None = None
        self._closing_reader: QuestReaderSession | None = None
        self._next_connect_at = 0.0
        self._connect_failure_reported = False
        self._last_sample_at: float | None = None
        self._last_sample_token: tuple[int, int] | None = None
        self._pose_publisher = self.create_publisher(PoseStamped, pose_topic, 10)
        self._joy_publisher = self.create_publisher(Joy, joy_topic, 10)
        self.create_timer(1.0 / poll_rate_hz, self._poll)
        self.get_logger().info(f"Meta Quest input bridge publishing {pose_topic} and {joy_topic}")

    def _poll(self) -> None:
        now = time.monotonic()
        if self._reader is None:
            if self._closing_reader is not None:
                if not self._closing_reader.close(timeout=0.0):
                    return
                self._closing_reader = None
            if now < self._next_connect_at:
                return
            self._next_connect_at = now + self._reconnect_period
            try:
                self._reader = self._reader_factory(self._ip_address, self._adb_port)
            except (Exception, SystemExit) as exc:
                if not self._connect_failure_reported:
                    self.get_logger().warning(
                        f"Meta Quest unavailable; waiting to reconnect: {exc}"
                    )
                    self._connect_failure_reported = True
                return
            self._connect_failure_reported = False
            now = time.monotonic()
            self._last_sample_at = now
            self._last_sample_token = None
            connection = self._ip_address or "USB"
            self.get_logger().info(f"Meta Quest connected through {connection}")

        try:
            transformations, buttons = self._reader.get_transformations_and_buttons()
        except Exception as exc:
            self._disconnect_reader(f"reader error: {exc}")
            return

        sample_token = (id(transformations), id(buttons))
        if sample_token == self._last_sample_token:
            if (
                self._last_sample_at is not None
                and now - self._last_sample_at > self._sample_timeout
            ):
                self._disconnect_reader("controller stream became stale")
            return
        self._last_sample_token = sample_token
        self._last_sample_at = now
        if not transformations or "r" not in transformations or not buttons:
            return

        stamp = self.get_clock().now().to_msg()
        transform = transformations["r"]
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self._frame_id
        pose.pose.position.x = float(transform[0][3])
        pose.pose.position.y = float(transform[1][3])
        pose.pose.position.z = float(transform[2][3])
        quaternion = _rotation_matrix_to_quaternion(transform)
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = quaternion
        self._pose_publisher.publish(pose)

        left_x, left_y = _joystick(buttons.get("leftJS"))
        right_x, right_y = _joystick(buttons.get("rightJS"))
        joy = Joy()
        joy.header.stamp = stamp
        joy.header.frame_id = self._frame_id
        joy.axes = [left_x, left_y, right_y, right_x]
        joy.buttons = [
            int(bool(buttons.get("A", False))),
            int(bool(buttons.get("B", False))),
            int(bool(buttons.get("X", False))),
            int(bool(buttons.get("Y", False))),
            int(bool(buttons.get("LG", False))),
            int(bool(buttons.get("RG", False))),
        ]
        self._joy_publisher.publish(joy)

    def _disconnect_reader(self, reason: str) -> None:
        if self._reader is not None and not self._reader.close(
            timeout=READER_CLOSE_TIMEOUT_SECONDS
        ):
            self._closing_reader = self._reader
        self._reader = None
        self._last_sample_at = None
        self._last_sample_token = None
        self._next_connect_at = time.monotonic() + self._reconnect_period
        self.get_logger().warning(f"Meta Quest disconnected; {reason}")

    def close(self) -> None:
        readers = (self._reader, self._closing_reader)
        self._reader = None
        self._closing_reader = None
        for reader in readers:
            if reader is not None and not reader.close(timeout=1.0):
                self.get_logger().warning("Meta Quest logcat worker did not stop before shutdown")


def _joystick(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    try:
        return float(value[0]), float(value[1])
    except (IndexError, TypeError, ValueError):
        return 0.0, 0.0


def _rotation_matrix_to_quaternion(matrix: Any) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation block to an x/y/z/w unit quaternion."""

    m00, m01, m02 = (float(matrix[0][index]) for index in range(3))
    m10, m11, m12 = (float(matrix[1][index]) for index in range(3))
    m20, m21, m22 = (float(matrix[2][index]) for index in range(3))
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)


def _load_meta_quest_reader(ip_address: str, adb_port: int) -> QuestReaderSession:
    target = f"{ip_address}:{adb_port}" if ip_address else None
    command = ["adb", "connect", target] if target else ["adb", "devices"]
    try:
        adb = subprocess.run(command, capture_output=True, check=False, text=True, timeout=5.0)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("ADB is unavailable") from exc
    if adb.returncode != 0:
        detail = (adb.stderr or adb.stdout).strip()
        raise RuntimeError(detail or "ADB could not connect to the Meta Quest")
    if target:
        if "connected" not in adb.stdout:
            raise RuntimeError(adb.stdout.strip() or f"ADB could not connect to {target}")
    elif not any("\tdevice" in line for line in adb.stdout.splitlines()):
        raise RuntimeError("no authorized USB Meta Quest was found by ADB")

    try:
        module = importlib.import_module("oculus_reader")
        reader_type = module.OculusReader
    except (AttributeError, ImportError):
        try:
            module = importlib.import_module("oculus_reader.reader")
            reader_type = module.OculusReader
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(
                "Install the maintained Meta Quest reader package before starting the "
                "Meta Quest hardware bridge."
            ) from exc
    reader = reader_type(ip_address=ip_address or None, port=adb_port, run=False)
    reader.running = True
    component = f"{reader.APK_name}/{reader.APK_name}.MainActivity"
    reader.device.shell(
        f'am start -n "{component}" -a android.intent.action.MAIN '
        "-c android.intent.category.LAUNCHER"
    )
    return QuestReaderSession(reader)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: MetaQuestBridge | None = None
    try:
        node = MetaQuestBridge(_load_meta_quest_reader)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
