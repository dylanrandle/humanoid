"""ROS 2 hardware-edge adapter for Meta Quest controller input."""

from __future__ import annotations

import importlib
import math
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Joy

from triskel_operator.topics import QUEST_JOY_TOPIC, QUEST_POSE_TOPIC


class MetaQuestBridge(Node):
    """Publish Meta Quest reader samples as standard ROS pose and joystick messages."""

    def __init__(self, reader: Any) -> None:
        super().__init__("meta_quest_bridge")
        self.declare_parameter("poll_rate_hz", 60.0)
        self.declare_parameter("frame_id", "openxr")
        self.declare_parameter("pose_topic", QUEST_POSE_TOPIC)
        self.declare_parameter("joy_topic", QUEST_JOY_TOPIC)

        poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        if not math.isfinite(poll_rate_hz) or poll_rate_hz <= 0.0:
            raise ValueError("poll_rate_hz must be positive and finite")
        self._frame_id = str(self.get_parameter("frame_id").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        joy_topic = str(self.get_parameter("joy_topic").value)
        self._reader = reader
        self._pose_publisher = self.create_publisher(PoseStamped, pose_topic, 10)
        self._joy_publisher = self.create_publisher(Joy, joy_topic, 10)
        self.create_timer(1.0 / poll_rate_hz, self._poll)
        self.get_logger().info(f"Meta Quest input bridge publishing {pose_topic} and {joy_topic}")

    def _poll(self) -> None:
        transformations, buttons = self._reader.get_transformations_and_buttons()
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


def _load_meta_quest_reader() -> Any:
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
    return reader_type()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: MetaQuestBridge | None = None
    try:
        node = MetaQuestBridge(_load_meta_quest_reader())
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
