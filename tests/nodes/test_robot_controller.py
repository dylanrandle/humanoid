from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pinocchio as pin
import pytest

from humanoid.constants import Topic
from humanoid.controllers.operational_space import ControlResult
from humanoid.nodes.robot_controller import RobotController
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotConfig,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.servo import ServoControlMode


def _make_robot_config() -> RobotConfig:
    return RobotConfig(
        name="panda",
        tool_frame="panda_hand",
        home_position=np.zeros(7),
        rest_position=np.ones(7),
        joint_idx_to_servo_id={i: i + 1 for i in range(7)},
        servo_control_modes=dict.fromkeys(range(1, 8), ServoControlMode.POSITION),
    )


def _make_controller(robot_config: RobotConfig | None = None) -> RobotController:
    """Build a RobotController with mocked LCM, Robot, and OSC."""
    if robot_config is None:
        robot_config = _make_robot_config()

    with (
        patch("humanoid.nodes.robot_controller.Subscriber"),
        patch("humanoid.nodes.robot_controller.Publisher"),
        patch("humanoid.nodes.robot_controller.Robot") as mock_robot_cls,
        patch("humanoid.nodes.robot_controller.OperationalSpaceController") as mock_osc_cls,
    ):
        mock_robot_cls.return_value = MagicMock()

        mock_osc = MagicMock()
        mock_osc.configuration = None
        mock_osc.compute_control.return_value = ControlResult(
            q=np.arange(7, dtype=float), v=np.arange(7, dtype=float) * 0.1
        )
        mock_osc_cls.return_value = mock_osc

        return RobotController(robot_config=robot_config)


def _no_messages(topic, timeout=0):
    return None


def _make_tool_cmd(gripper=None):
    return RobotToolCommand(
        timestamp=0.0,
        pose=pin.SE3(np.eye(3), np.array([0.3, 0.0, 0.4])),
        gripper_positions=gripper,
    )


def _make_base_cmd():
    return RobotBaseCommand(
        timestamp=0.0,
        pose=pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])),
    )


def _make_state():
    return RobotState(
        timestamp=0.0,
        joint_positions=np.arange(7, dtype=float),
        joint_velocities=np.zeros(7),
        motor_temperatures=np.zeros(7),
    )


@pytest.fixture
def controller():
    return _make_controller()


def test_no_messages_does_nothing(controller):
    """When no commands arrive, no joint command is published."""
    controller.subscriber.receive = Mock(side_effect=_no_messages)

    controller.receive_and_compute()

    controller.publisher.publish.assert_not_called()
    controller.controller.compute_control.assert_not_called()
    controller.controller.update_state.assert_not_called()


def test_first_state_initializes_controller(controller):
    """The first RobotState message triggers update_state on the OSC."""
    state = _make_state()

    def receive(topic, timeout=0):
        if topic == Topic.ROBOT_STATE:
            return state
        return None

    controller.subscriber.receive = Mock(side_effect=receive)
    controller.receive_and_compute()

    controller.controller.update_state.assert_called_once()
    np.testing.assert_allclose(
        controller.controller.update_state.call_args[0][0], state.joint_positions
    )


def test_state_after_initialization_does_not_reinitialize(controller):
    """Once configuration is set, subsequent RobotState messages don't re-init."""
    controller.controller.configuration = MagicMock()  # already initialized

    def receive(topic, timeout=0):
        if topic == Topic.ROBOT_STATE:
            return _make_state()
        return None

    controller.subscriber.receive = Mock(side_effect=receive)
    controller.receive_and_compute()

    controller.controller.update_state.assert_not_called()


def test_tool_command_publishes_joint_command(controller):
    """A tool command triggers compute_control and publishes the result."""
    tool_cmd = _make_tool_cmd()

    def receive(topic, timeout=0):
        if topic == Topic.ROBOT_TOOL_COMMAND:
            return tool_cmd
        return None

    controller.subscriber.receive = Mock(side_effect=receive)
    controller.receive_and_compute()

    controller.controller.compute_control.assert_called_once()
    call_kwargs = controller.controller.compute_control.call_args
    # First positional arg is the tool pose.
    assert call_kwargs.args[0] is tool_cmd.pose
    assert call_kwargs.kwargs["base_target_pose"] is None
    assert call_kwargs.kwargs["gripper_positions"] is None

    controller.publisher.publish.assert_called_once()
    published = controller.publisher.publish.call_args[0][0]
    assert isinstance(published, RobotJointCommand)
    np.testing.assert_allclose(published.joint_positions, np.arange(7, dtype=float))
    np.testing.assert_allclose(published.joint_velocities, np.arange(7, dtype=float) * 0.1)


def test_base_command_alone_does_not_publish(controller):
    """A base command without a tool command does not trigger control."""
    base_cmd = _make_base_cmd()

    def receive(topic, timeout=0):
        if topic == Topic.ROBOT_BASE_COMMAND:
            return base_cmd
        return None

    controller.subscriber.receive = Mock(side_effect=receive)
    controller.receive_and_compute()

    # Base command is stored, but no compute/publish happens without a tool command.
    assert controller.current_base_command is base_cmd
    controller.controller.compute_control.assert_not_called()
    controller.publisher.publish.assert_not_called()


def test_tool_command_persists_across_ticks(controller):
    """Once received, the tool command keeps driving control on subsequent ticks."""
    tool_cmd = _make_tool_cmd()
    call_count = {"n": 0}

    def receive(topic, timeout=0):
        if topic == Topic.ROBOT_TOOL_COMMAND and call_count["n"] == 0:
            return tool_cmd
        return None

    controller.subscriber.receive = Mock(side_effect=receive)

    controller.receive_and_compute()  # delivers tool command
    call_count["n"] += 1
    controller.receive_and_compute()  # no new messages, but should still publish

    expected_call_count = 2

    assert controller.controller.compute_control.call_count == expected_call_count
    assert controller.publisher.publish.call_count == expected_call_count


def test_base_command_passed_to_compute_control(controller):
    """When both tool and base commands have been received, OSC gets both poses."""
    tool_cmd = _make_tool_cmd()
    base_cmd = _make_base_cmd()

    def receive(topic, timeout=0):
        if topic == Topic.ROBOT_TOOL_COMMAND:
            return tool_cmd
        if topic == Topic.ROBOT_BASE_COMMAND:
            return base_cmd
        return None

    controller.subscriber.receive = Mock(side_effect=receive)
    controller.receive_and_compute()

    call = controller.controller.compute_control.call_args
    assert call.args[0] is tool_cmd.pose
    assert call.kwargs["base_target_pose"] is base_cmd.pose


def test_gripper_positions_passed_through(controller):
    """Gripper positions on the tool command are forwarded to compute_control."""
    gripper = np.array([0.01, 0.02])
    tool_cmd = _make_tool_cmd(gripper=gripper)

    def receive(topic, timeout=0):
        if topic == Topic.ROBOT_TOOL_COMMAND:
            return tool_cmd
        return None

    controller.subscriber.receive = Mock(side_effect=receive)
    controller.receive_and_compute()

    np.testing.assert_allclose(
        controller.controller.compute_control.call_args.kwargs["gripper_positions"], gripper
    )


def test_close_closes_subscriber(controller):
    controller.close()
    controller.subscriber.close.assert_called_once()
