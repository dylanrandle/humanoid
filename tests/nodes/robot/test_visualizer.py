from unittest.mock import MagicMock, patch

import numpy as np
import pinocchio as pin

from humanoid.config import ROBOT_CONFIGS
from humanoid.constants import Topic
from humanoid.nodes.robot.visualizer import RobotVisualizerNode
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.visualizer import VisualizerConfig


def _make_node(robot_name: str):
    with (
        patch("humanoid.nodes.robot.visualizer.Robot") as robot_class,
        patch("humanoid.nodes.robot.visualizer._RobotVisualizer") as visualizer_class,
        patch("humanoid.nodes.robot.visualizer.Subscriber") as subscriber_class,
    ):
        robot = MagicMock()
        robot_class.return_value = robot
        visualizer = MagicMock()
        visualizer_class.return_value = visualizer
        subscriber = MagicMock()
        subscriber_class.return_value = subscriber
        node = RobotVisualizerNode(
            robot_config=ROBOT_CONFIGS[robot_name],
            visualizer_config=VisualizerConfig(),
        )

    visualizer.reset_mock()
    return node, robot, visualizer, subscriber


def _robot_state(q: np.ndarray) -> RobotState:
    return RobotState(
        timestamp=0.0,
        joint_positions=q,
        joint_velocities=np.zeros_like(q),
        actuator_temperatures=np.zeros_like(q),
    )


def _set_messages(subscriber: MagicMock, messages: dict[Topic, object]) -> None:
    subscriber.receive.side_effect = messages.get


def _assert_se3_equal(actual: pin.SE3, expected: pin.SE3) -> None:
    np.testing.assert_allclose(actual.rotation, expected.rotation, atol=1e-12)
    np.testing.assert_allclose(actual.translation, expected.translation, atol=1e-12)


def test_fixed_base_tool_command_is_already_in_world_frame():
    node, robot, visualizer, subscriber = _make_node("panda")
    q = ROBOT_CONFIGS["panda"].homing_presets[HomingPreset.HOME].copy()
    tool_command_pose = pin.SE3(
        pin.utils.rotate("z", 0.3),
        np.array([0.4, -0.2, 0.7]),
    )
    robot.get_base_pose.return_value = None
    _set_messages(
        subscriber,
        {
            Topic.ROBOT_STATE: _robot_state(q),
            Topic.ROBOT_TOOL_COMMAND: RobotToolCommand(
                timestamp=0.0,
                pose=tool_command_pose,
            ),
        },
    )

    node.step()

    displayed_pose = visualizer.display_tool_command.call_args.args[0]
    _assert_se3_equal(displayed_pose, tool_command_pose)
    np.testing.assert_array_equal(robot.get_base_pose.call_args.args[0], q)


def test_mobile_tool_command_uses_measured_not_commanded_base_pose():
    node, robot, visualizer, subscriber = _make_node("elrobot_mobile")
    q = ROBOT_CONFIGS["elrobot_mobile"].homing_presets[HomingPreset.HOME].copy()
    measured_base_pose = pin.SE3(
        pin.utils.rotate("z", np.pi / 2),
        np.array([1.0, 2.0, 0.0]),
    )
    commanded_base_pose = pin.SE3(
        pin.utils.rotate("z", -0.4),
        np.array([10.0, -5.0, 0.0]),
    )
    relative_tool_pose = pin.SE3(
        pin.utils.rotate("x", 0.2),
        np.array([0.4, -0.1, 0.6]),
    )
    robot.get_base_pose.return_value = measured_base_pose
    _set_messages(
        subscriber,
        {
            Topic.ROBOT_STATE: _robot_state(q),
            Topic.ROBOT_TOOL_COMMAND: RobotToolCommand(
                timestamp=0.0,
                pose=relative_tool_pose,
            ),
            Topic.ROBOT_BASE_COMMAND: RobotBaseCommand(
                timestamp=0.0,
                pose=commanded_base_pose,
            ),
        },
    )

    node.step()

    displayed_tool_pose = visualizer.display_tool_command.call_args.args[0]
    _assert_se3_equal(displayed_tool_pose, measured_base_pose * relative_tool_pose)
    displayed_base_pose = visualizer.display_base_command.call_args.args[0]
    _assert_se3_equal(displayed_base_pose, commanded_base_pose)
    np.testing.assert_array_equal(robot.get_base_pose.call_args.args[0], q)
