from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.constants import Topic
from humanoid.controllers.operational_space import ControlResult
from humanoid.nodes.robot.controller import RobotControllerNode
from humanoid.types.actuator import (
    ActuatorControlMode,
)
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
        patch("humanoid.nodes.robot.controller.OmniwheelBaseController") as mock_base_cls,
        patch("humanoid.nodes.robot.controller.GripperController") as mock_gripper_cls,
    ):
        is_mobile = robot_config.base is not None
        nq = robot_config.homing_presets[HomingPreset.HOME].shape[0]
        nv = 14 if is_mobile else nq
        mock_robot = MagicMock()
        mock_robot.model.nq = nq
        mock_robot.model.nv = nv
        # Geometry helpers used by _reset_commands_from_state.
        gripper_q_indices = [nq - 1] if robot_config.gripper_joint_indices else []
        mock_robot.get_gripper_position_indices.return_value = gripper_q_indices
        mock_robot.get_tool_command_pose.return_value = pin.SE3.Identity()
        mock_robot.get_base_pose.return_value = pin.SE3.Identity() if is_mobile else None
        mock_robot_cls.return_value = mock_robot

        mock_osc = MagicMock()
        mock_osc.configuration = None
        if is_mobile:
            mock_osc.controlled_q_indices = np.arange(10, 17)
            mock_osc.controlled_v_indices = np.arange(6, 13)
        else:
            mock_osc.controlled_q_indices = np.arange(nq)
            mock_osc.controlled_v_indices = np.arange(nv)
        mock_osc.compute_control.return_value = ControlResult(
            q=np.arange(nq, dtype=float), v=np.arange(nv, dtype=float) * 0.1
        )
        mock_osc_cls.return_value = mock_osc

        mock_base = MagicMock()
        mock_base.configuration = None
        mock_base.controlled_q_indices = np.arange(10)
        mock_base.controlled_v_indices = np.arange(6)
        mock_base.compute_control.return_value = ControlResult(
            q=np.arange(nq, dtype=float) + 100.0,
            v=np.arange(nv, dtype=float) + 100.0,
        )
        mock_base_cls.return_value = mock_base

        mock_gripper = MagicMock()
        mock_gripper.controlled_q_indices = np.asarray(gripper_q_indices)
        mock_gripper.controlled_v_indices = np.array([nv - 1], dtype=int)
        gripper_state = {"q": np.zeros(nq)}

        def update_gripper_state(q):
            gripper_state["q"] = q.copy()

        def compute_gripper_control(positions):
            q = gripper_state["q"].copy()
            q[gripper_q_indices] = positions
            return ControlResult(q=q, v=np.zeros(nv))

        mock_gripper.update_state.side_effect = update_gripper_state
        mock_gripper.compute_control.side_effect = compute_gripper_control
        mock_gripper_cls.return_value = mock_gripper

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


def _make_state(nq=7, nv=None):
    if nv is None:
        nv = nq
    return RobotState(
        timestamp=0.0,
        joint_positions=np.arange(nq, dtype=float),
        joint_velocities=np.zeros(nv),
        actuator_temperatures=np.zeros(nq),
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


@pytest.fixture
def active_mobile_controller():
    c = _make_controller(TRISKEL_CONFIG)
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
        """Once commanded, feedback does not replace the open-loop arm state."""
        active_controller.controller.configuration = MagicMock()  # already initialized
        active_controller.current_tool_command = _make_tool_cmd()
        state = _make_state()
        state.joint_positions += 100.0

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_STATE:
                return state
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()

        active_controller.controller.update_state.assert_called_once()
        np.testing.assert_allclose(
            active_controller.controller.update_state.call_args.args[0],
            np.arange(7, dtype=float),
        )

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
        assert not call_kwargs.kwargs

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

    def test_base_command_alone_drives_mobile_base(self, active_mobile_controller):
        base_cmd = _make_base_cmd()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_BASE_COMMAND:
                return base_cmd
            return None

        active_mobile_controller.subscriber.receive = Mock(side_effect=receive)
        active_mobile_controller.step()

        active_mobile_controller.base_controller.compute_control.assert_called_once_with(
            base_cmd.pose
        )
        active_mobile_controller.arm_controller.compute_control.assert_not_called()
        published = active_mobile_controller.publisher.publish.call_args.args[0]
        np.testing.assert_allclose(published.joint_velocities[:6], np.arange(6) + 100.0)

    def test_arm_and_base_commands_use_separate_controllers(self, active_mobile_controller):
        tool_cmd = _make_tool_cmd()
        base_cmd = _make_base_cmd()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_TOOL_COMMAND:
                return tool_cmd
            if topic == Topic.ROBOT_BASE_COMMAND:
                return base_cmd
            return None

        active_mobile_controller.subscriber.receive = Mock(side_effect=receive)
        active_mobile_controller.step()

        active_mobile_controller.arm_controller.compute_control.assert_called_once_with(
            tool_cmd.pose
        )
        active_mobile_controller.base_controller.compute_control.assert_called_once_with(
            base_cmd.pose
        )

    def test_gripper_positions_use_separate_controller(self, active_mobile_controller):
        gripper = np.array([0.01])
        tool_cmd = _make_tool_cmd(gripper=gripper)
        arm_result = active_mobile_controller.arm_controller.compute_control.return_value
        osc_gripper_positions = []

        def compute_arm_control(tool_pose):
            state = active_mobile_controller.arm_controller.update_state.call_args.args[0]
            osc_gripper_positions.append(state[-1])
            return arm_result

        active_mobile_controller.arm_controller.compute_control.side_effect = compute_arm_control

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_TOOL_COMMAND:
                return tool_cmd
            return None

        active_mobile_controller.subscriber.receive = Mock(side_effect=receive)
        active_mobile_controller.step()

        active_mobile_controller.gripper_controller.compute_control.assert_called_once_with(gripper)
        assert osc_gripper_positions == [pytest.approx(gripper[0])]
        arm_call = active_mobile_controller.arm_controller.compute_control.call_args
        assert arm_call.args == (tool_cmd.pose,)
        published = active_mobile_controller.publisher.publish.call_args.args[0]
        assert published.joint_positions[-1] == pytest.approx(gripper[0])


def test_close_closes_subscriber(controller):
    controller.close()
    controller.subscriber.close.assert_called_once()
