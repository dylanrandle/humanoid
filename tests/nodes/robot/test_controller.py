from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pinocchio as pin
import pytest

from humanoid.constants import Topic
from humanoid.controllers.operational_space import ControlResult
from humanoid.hardware.actuators.config import (
    ActuatorControlMode,
)
from humanoid.nodes.robot.controller import RobotControllerNode
from humanoid.types.homing import HomingPreset
from humanoid.types.orchestrator import Mode, OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotConfig,
    RobotJointCommand,
    RobotName,
    RobotState,
    RobotToolCommand,
    RobotToolConfig,
)


def _make_robot_config() -> RobotConfig:
    return RobotConfig(
        name=RobotName.PANDA,
        tool=RobotToolConfig(frame="panda_hand"),
        homing_presets={
            HomingPreset.HOME: np.zeros(7),
            HomingPreset.REST: np.ones(7),
        },
        actuator_control_modes={f"joint_{i}": ActuatorControlMode.POSITION for i in range(7)},
        hardware=None,
    )


def _make_controller(robot_config: RobotConfig | None = None) -> RobotControllerNode:
    """Build a RobotControllerNode with mocked LCM, Robot, and OSC."""
    if robot_config is None:
        robot_config = _make_robot_config()

    with (
        patch("humanoid.nodes.robot.controller.Subscriber"),
        patch("humanoid.nodes.robot.controller.Publisher"),
        patch("humanoid.nodes.robot.controller.Robot") as mock_robot_cls,
        patch("humanoid.nodes.robot.controller.OperationalSpaceController") as mock_osc_cls,
    ):
        mock_robot = MagicMock()
        # Geometry helpers used by _reset_commands_from_state.
        mock_robot.get_gripper_position_indices.return_value = []
        mock_robot.get_tool_command_pose.return_value = pin.SE3.Identity()
        mock_robot.get_base_pose.return_value = None
        mock_robot_cls.return_value = mock_robot

        mock_osc = MagicMock()
        mock_osc.configuration = None
        mock_osc.compute_control.return_value = ControlResult(
            q=np.arange(7, dtype=float), v=np.arange(7, dtype=float) * 0.1
        )
        mock_osc_cls.return_value = mock_osc

        return RobotControllerNode(robot_config=robot_config)


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
        actuator_temperatures=np.zeros(7),
    )


def _activate(controller: RobotControllerNode) -> None:
    """Put the controller into an active mode so the OSC compute path runs."""
    controller.current_mode = Mode.OCULUS


@pytest.fixture
def controller():
    return _make_controller()


@pytest.fixture
def active_controller():
    c = _make_controller()
    _activate(c)
    return c


def test_no_messages_does_nothing(controller):
    """With no orchestrator messages and no state, no work happens."""
    controller.subscriber.receive = Mock(side_effect=_no_messages)

    controller.step()

    controller.publisher.publish.assert_not_called()
    controller.controller.compute_control.assert_not_called()
    controller.controller.update_state.assert_not_called()


class TestInactiveMode:
    """In inactive modes the OSC re-syncs from state but never publishes."""

    def test_state_drives_update_state_every_step(self, controller):
        """While inactive, every received state re-syncs the OSC."""
        state = _make_state()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_STATE:
                return state
            return None

        controller.subscriber.receive = Mock(side_effect=receive)
        controller.step()
        controller.step()

        expected = 2
        assert controller.controller.update_state.call_count == expected

    def test_inactive_resets_tool_command_from_fk(self, controller):
        """Inactive mode rewrites current_tool_command from FK each step."""
        controller.current_tool_command = _make_tool_cmd()  # stale target from before

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_STATE:
                return _make_state()
            return None

        controller.subscriber.receive = Mock(side_effect=receive)
        controller.step()

        # Tool pose now comes from FK on the current state, not the stale target.
        controller.robot.get_tool_command_pose.assert_called_once()
        assert (
            controller.current_tool_command.pose
            is controller.robot.get_tool_command_pose.return_value
        )

    def test_inactive_does_not_publish(self, controller):
        """Even if a tool command arrives, inactive mode publishes nothing."""

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_STATE:
                return _make_state()
            if topic == Topic.ROBOT_TOOL_COMMAND:
                return _make_tool_cmd()
            return None

        controller.subscriber.receive = Mock(side_effect=receive)
        controller.step()

        controller.publisher.publish.assert_not_called()
        controller.controller.compute_control.assert_not_called()


class TestModeTransitions:
    def test_orchestrator_mode_message_updates_current_mode(self, controller):
        assert controller.current_mode is Mode.IDLE

        def receive(topic, timeout=0):
            if topic == Topic.ORCHESTRATOR_MODE:
                return OrchestratorMode(timestamp=0.0, mode=Mode.OCULUS)
            return None

        controller.subscriber.receive = Mock(side_effect=receive)
        controller.step()

        assert controller.current_mode is Mode.OCULUS
        assert controller.is_active

    def test_homing_mode_keeps_controller_inactive(self, controller):
        def receive(topic, timeout=0):
            if topic == Topic.ORCHESTRATOR_MODE:
                return OrchestratorMode(timestamp=0.0, mode=Mode.HOMING)
            if topic == Topic.ROBOT_STATE:
                return _make_state()
            return None

        controller.subscriber.receive = Mock(side_effect=receive)
        controller.step()

        assert controller.current_mode is Mode.HOMING
        assert not controller.is_active
        controller.publisher.publish.assert_not_called()


class TestActiveMode:
    def test_first_state_initializes_controller(self, active_controller):
        """The first RobotState message triggers update_state on the OSC."""
        state = _make_state()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_STATE:
                return state
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()

        active_controller.controller.update_state.assert_called_once()
        np.testing.assert_allclose(
            active_controller.controller.update_state.call_args[0][0], state.joint_positions
        )

    def test_state_updates_continuously_until_first_tool_command(self, active_controller):
        """RobotState keeps driving update_state until a tool command arrives."""
        active_controller.controller.configuration = MagicMock()  # already initialized

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_STATE:
                return _make_state()
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()
        active_controller.step()

        expected_update_calls = 2
        assert active_controller.controller.update_state.call_count == expected_update_calls

    def test_state_does_not_update_after_tool_command(self, active_controller):
        """Once a tool command has been received, RobotState no longer re-inits."""
        active_controller.controller.configuration = MagicMock()  # already initialized
        active_controller.current_tool_command = _make_tool_cmd()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_STATE:
                return _make_state()
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()

        active_controller.controller.update_state.assert_not_called()

    def test_tool_command_publishes_joint_command_on_osc_topic(self, active_controller):
        """A tool command triggers compute_control and publishes to OSC_JOINT_COMMAND."""
        tool_cmd = _make_tool_cmd()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_TOOL_COMMAND:
                return tool_cmd
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()

        active_controller.controller.compute_control.assert_called_once()
        call_kwargs = active_controller.controller.compute_control.call_args
        # First positional arg is the tool pose.
        assert call_kwargs.args[0] is tool_cmd.pose
        assert call_kwargs.kwargs["base_target_pose"] is None
        assert call_kwargs.kwargs["gripper_positions"] is None

        active_controller.publisher.publish.assert_called_once()
        published = active_controller.publisher.publish.call_args
        assert isinstance(published.args[0], RobotJointCommand)
        np.testing.assert_allclose(published.args[0].joint_positions, np.arange(7, dtype=float))
        np.testing.assert_allclose(
            published.args[0].joint_velocities, np.arange(7, dtype=float) * 0.1
        )
        assert published.kwargs["topic"] is Topic.CONTROLLER_JOINT_COMMAND

    def test_base_command_alone_does_not_publish(self, active_controller):
        """A base command without a tool command does not trigger control."""
        base_cmd = _make_base_cmd()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_BASE_COMMAND:
                return base_cmd
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()

        # Base command is stored, but no compute/publish happens without a tool command.
        assert active_controller.current_base_command is base_cmd
        active_controller.controller.compute_control.assert_not_called()
        active_controller.publisher.publish.assert_not_called()

    def test_tool_command_persists_across_ticks(self, active_controller):
        """Once received, the tool command keeps driving control on subsequent ticks."""
        tool_cmd = _make_tool_cmd()
        call_count = {"n": 0}

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_TOOL_COMMAND and call_count["n"] == 0:
                return tool_cmd
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)

        active_controller.step()  # delivers tool command
        call_count["n"] += 1
        active_controller.step()  # no new messages, but should still publish

        expected_call_count = 2

        assert active_controller.controller.compute_control.call_count == expected_call_count
        assert active_controller.publisher.publish.call_count == expected_call_count

    def test_base_command_passed_to_compute_control(self, active_controller):
        """When both tool and base commands have been received, OSC gets both poses."""
        tool_cmd = _make_tool_cmd()
        base_cmd = _make_base_cmd()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_TOOL_COMMAND:
                return tool_cmd
            if topic == Topic.ROBOT_BASE_COMMAND:
                return base_cmd
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()

        call = active_controller.controller.compute_control.call_args
        assert call.args[0] is tool_cmd.pose
        assert call.kwargs["base_target_pose"] is base_cmd.pose

    def test_gripper_positions_passed_through(self, active_controller):
        """Gripper positions on the tool command are forwarded to compute_control."""
        gripper = np.array([0.01, 0.02])
        tool_cmd = _make_tool_cmd(gripper=gripper)

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_TOOL_COMMAND:
                return tool_cmd
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()

        np.testing.assert_allclose(
            active_controller.controller.compute_control.call_args.kwargs["gripper_positions"],
            gripper,
        )


def test_close_closes_subscriber(controller):
    controller.close()
    controller.subscriber.close.assert_called_once()
