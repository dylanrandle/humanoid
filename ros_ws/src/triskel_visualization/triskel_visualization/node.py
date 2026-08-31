"""ROS node serving a layered browser-native Viser view of Triskel."""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import viser
import yourdfpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory
from viser.extras import ViserUrdf

from triskel_visualization.model import write_resolved_urdf

READY_TOPIC = "/triskel/visualization/ready"
ARM_COMMAND_TOPIC = "/arm_controller/joint_trajectory"
GRIPPER_COMMAND_TOPIC = "/gripper_controller/joint_trajectory"
TOOL_COMMAND_TOPIC = "/servo_node/delta_twist_cmds"
JOINT_STATE_TOPIC = "/joint_states"
ODOMETRY_TOPIC = "/omni_base_controller/odom"
BASE_FRAME = "base_link"
END_EFFECTOR_FRAME = "gripper_base_link"
QUATERNION_MIN_NORM = 1e-9
ROTATION_EPSILON = 1e-12
CONTROL_PERIOD_SECONDS = 0.05
MAX_INTEGRATION_PERIOD_SECONDS = 0.1
TOOL_COMMAND_TIMEOUT_SECONDS = 0.3
TASK_TARGET_COLOR = (255, 145, 55)
COMMAND_GHOST_COLOR = (105, 255, 125, 0.28)


def _load_urdf(path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        path,
        load_meshes=True,
        build_scene_graph=True,
        load_collision_meshes=False,
        build_collision_scene_graph=False,
    )


def _matrix_to_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            float(matrix[2, 1] - matrix[1, 2]) / scale,
            float(matrix[0, 2] - matrix[2, 0]) / scale,
            float(matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = (
                float(matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                float(matrix[0, 1] + matrix[1, 0]) / scale,
                float(matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = (
                float(matrix[0, 2] - matrix[2, 0]) / scale,
                float(matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                float(matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = (
                float(matrix[1, 0] - matrix[0, 1]) / scale,
                float(matrix[0, 2] + matrix[2, 0]) / scale,
                float(matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def _rotation_increment(vector: np.ndarray) -> np.ndarray:
    """Return the Rodrigues rotation for an angular displacement vector."""

    angle = float(np.linalg.norm(vector))
    if angle < ROTATION_EPSILON:
        return np.eye(3)
    axis = vector / angle
    cross = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _is_descendant(scene: Any, node_name: str, ancestor: str) -> bool:
    current = node_name
    while current != scene.graph.base_frame:
        if current == ancestor:
            return True
        parents = scene.graph.transforms.parents
        if current not in parents:
            return False
        current = parents[current]
    return current == ancestor


class TaskSpaceGripper:
    """Render only the gripper subtree at the unconstrained task-space target."""

    def __init__(
        self,
        server: viser.ViserServer,
        urdf: yourdfpy.URDF,
        root_node_name: str,
    ) -> None:
        self._urdf = urdf
        self._root = server.scene.add_frame(root_node_name, show_axes=False)
        self._meshes: list[tuple[str, Any]] = []
        scene = urdf.scene
        for node_name in scene.graph.nodes_geometry:
            if not _is_descendant(scene, node_name, END_EFFECTOR_FRAME):
                continue
            transform, geometry_name = scene.graph.get(
                frame_to=node_name,
                frame_from=END_EFFECTOR_FRAME,
            )
            geometry = scene.geometry[geometry_name]
            handle = server.scene.add_mesh_simple(
                f"{root_node_name}/geometry/{node_name}",
                vertices=np.asarray(geometry.vertices),
                faces=np.asarray(geometry.faces),
                color=TASK_TARGET_COLOR,
                opacity=0.72,
            )
            handle.position = tuple(float(value) for value in transform[:3, 3])
            handle.wxyz = _matrix_to_wxyz(transform[:3, :3])
            self._meshes.append((node_name, handle))

    @property
    def position(self) -> tuple[float, float, float]:
        return self._root.position

    @position.setter
    def position(self, value: tuple[float, float, float]) -> None:
        self._root.position = value

    @property
    def wxyz(self) -> tuple[float, float, float, float]:
        return self._root.wxyz

    @wxyz.setter
    def wxyz(self, value: tuple[float, float, float, float]) -> None:
        self._root.wxyz = value

    def update_cfg(self, configuration: dict[str, float]) -> None:
        self._urdf.update_cfg(configuration)
        for node_name, handle in self._meshes:
            transform, _ = self._urdf.scene.graph.get(
                frame_to=node_name,
                frame_from=END_EFFECTOR_FRAME,
            )
            handle.position = tuple(float(value) for value in transform[:3, 3])
            handle.wxyz = _matrix_to_wxyz(transform[:3, :3])


class TriskelVisualizer(Node):
    """Render measured, controller-commanded, and task-target robot state."""

    def __init__(self) -> None:
        super().__init__("triskel_visualizer")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8080)

        description_share = Path(get_package_share_directory("triskel_description"))
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="triskel-viser-")
        resolved_urdf = write_resolved_urdf(
            description_share / "urdf" / "triskel.urdf",
            description_share,
            Path(self._temporary_directory.name) / "triskel.urdf",
        )
        measured_urdf = _load_urdf(resolved_urdf)
        commanded_urdf = _load_urdf(resolved_urdf)
        task_gripper_urdf = _load_urdf(resolved_urdf)

        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        self._server = viser.ViserServer(host=host, port=port, label="Triskel", verbose=False)
        self._server.scene.set_up_direction((0.0, 0.0, 1.0))
        self._server.scene.world_axes.visible = False
        self._server.initial_camera.position = (1.0, -1.0, 0.75)
        self._server.initial_camera.look_at = (0.0, 0.0, 0.25)
        self._server.scene.add_grid(
            "/world/grid",
            width=4.0,
            height=4.0,
            plane="xy",
            cell_size=0.1,
            section_size=0.5,
        )
        self._base_frame = self._server.scene.add_frame("/triskel_base", show_axes=False)
        self._measured_visualizer = ViserUrdf(
            self._server,
            urdf_or_path=measured_urdf,
            root_node_name="/triskel_base/measured",
            load_meshes=True,
            load_collision_meshes=False,
        )
        self._commanded_visualizer = ViserUrdf(
            self._server,
            urdf_or_path=commanded_urdf,
            root_node_name="/triskel_base/commanded",
            load_meshes=True,
            load_collision_meshes=False,
            mesh_color_override=COMMAND_GHOST_COLOR,
        )
        self._task_gripper = TaskSpaceGripper(
            self._server,
            task_gripper_urdf,
            "/triskel_base/task_target",
        )

        self._joint_names = tuple(self._measured_visualizer.get_actuated_joint_names())
        self._joint_indices = {name: index for index, name in enumerate(self._joint_names)}
        self._measured_configuration = np.zeros(len(self._joint_names), dtype=float)
        self._commanded_configuration = np.zeros(len(self._joint_names), dtype=float)
        self._commanded_joints: set[str] = set()
        self._target_position = np.zeros(3)
        self._target_rotation = np.eye(3)
        self._target_initialized = False
        self._last_tool_command_at: float | None = None
        self._measured_urdf = measured_urdf
        self._measured_visualizer.update_cfg(self._measured_configuration)
        self._commanded_visualizer.update_cfg(self._commanded_configuration)
        self._task_gripper.update_cfg(self._configuration_dict(self._commanded_configuration))

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_state_callback, 10)
        self.create_subscription(JointTrajectory, ARM_COMMAND_TOPIC, self._arm_command_callback, 10)
        self.create_subscription(
            JointTrajectory,
            GRIPPER_COMMAND_TOPIC,
            self._gripper_command_callback,
            10,
        )
        self.create_subscription(TwistStamped, TOOL_COMMAND_TOPIC, self._tool_command_callback, 10)
        self.create_subscription(
            Odometry,
            ODOMETRY_TOPIC,
            self._odometry_callback,
            10,
        )
        self._ready_publisher = self.create_publisher(Bool, READY_TOPIC, 10)
        self.create_timer(1.0, self._publish_ready)
        self._publish_ready()
        self.get_logger().info(f"Triskel visualization available at http://{host}:{port}")

    def _configuration_dict(self, values: np.ndarray) -> dict[str, float]:
        return dict(zip(self._joint_names, values, strict=True))

    def _joint_state_callback(self, message: JointState) -> None:
        changed = False
        for name, position in zip(message.name, message.position, strict=False):
            index = self._joint_indices.get(name)
            if index is None or not np.isfinite(position):
                continue
            self._measured_configuration[index] = float(position)
            if name not in self._commanded_joints:
                self._commanded_configuration[index] = float(position)
            changed = True
        if not changed:
            return
        self._measured_visualizer.update_cfg(self._measured_configuration)
        self._commanded_visualizer.update_cfg(self._commanded_configuration)
        self._task_gripper.update_cfg(self._configuration_dict(self._commanded_configuration))
        now = time.monotonic()
        if (
            self._last_tool_command_at is None
            or now - self._last_tool_command_at > TOOL_COMMAND_TIMEOUT_SECONDS
        ):
            self._reset_task_target_from_measured()

    def _arm_command_callback(self, message: JointTrajectory) -> None:
        self._trajectory_command_callback(message)

    def _gripper_command_callback(self, message: JointTrajectory) -> None:
        self._trajectory_command_callback(message)

    def _trajectory_command_callback(self, message: JointTrajectory) -> None:
        if not message.points:
            return
        point = message.points[-1]
        changed = False
        for name, position in zip(message.joint_names, point.positions, strict=False):
            index = self._joint_indices.get(name)
            if index is None or not np.isfinite(position):
                continue
            self._commanded_configuration[index] = float(position)
            self._commanded_joints.add(name)
            changed = True
        if changed:
            self._commanded_visualizer.update_cfg(self._commanded_configuration)
            self._task_gripper.update_cfg(self._configuration_dict(self._commanded_configuration))

    def _tool_command_callback(self, message: TwistStamped) -> None:
        if not self._target_initialized:
            return
        now = time.monotonic()
        command_stream_stale = (
            self._last_tool_command_at is None
            or now - self._last_tool_command_at > TOOL_COMMAND_TIMEOUT_SECONDS
        )
        if command_stream_stale:
            self._reset_task_target_from_measured()
            elapsed = CONTROL_PERIOD_SECONDS
        else:
            elapsed = min(now - self._last_tool_command_at, MAX_INTEGRATION_PERIOD_SECONDS)
        self._last_tool_command_at = now
        linear = np.array(
            (message.twist.linear.x, message.twist.linear.y, message.twist.linear.z),
            dtype=float,
        )
        angular = np.array(
            (message.twist.angular.x, message.twist.angular.y, message.twist.angular.z),
            dtype=float,
        )
        if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(angular)):
            return
        self._target_position += linear * elapsed
        self._target_rotation = _rotation_increment(angular * elapsed) @ self._target_rotation
        self._update_task_target()

    def _reset_task_target_from_measured(self) -> None:
        transform = self._measured_urdf.get_transform(END_EFFECTOR_FRAME, BASE_FRAME)
        self._target_position = np.array(transform[:3, 3], dtype=float, copy=True)
        self._target_rotation = np.array(transform[:3, :3], dtype=float, copy=True)
        self._target_initialized = True
        self._update_task_target()

    def _update_task_target(self) -> None:
        self._task_gripper.position = tuple(float(value) for value in self._target_position)
        self._task_gripper.wxyz = _matrix_to_wxyz(self._target_rotation)

    def _odometry_callback(self, message: Odometry) -> None:
        pose = message.pose.pose
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(np.isfinite(value) for value in values):
            return
        quaternion_norm = float(np.linalg.norm(values[3:]))
        if quaternion_norm < QUATERNION_MIN_NORM:
            return
        self._base_frame.position = tuple(float(value) for value in values[:3])
        self._base_frame.wxyz = (
            float(values[6] / quaternion_norm),
            float(values[3] / quaternion_norm),
            float(values[4] / quaternion_norm),
            float(values[5] / quaternion_norm),
        )

    def _publish_ready(self) -> None:
        message = Bool()
        message.data = True
        self._ready_publisher.publish(message)

    def close(self) -> None:
        self._server.stop()
        self._temporary_directory.cleanup()


def main() -> None:
    rclpy.init()
    node = TriskelVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
