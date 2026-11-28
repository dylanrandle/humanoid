from typing import cast
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
    JointDriveConfig,
)
from humanoid.types.homing import HomingPreset
from humanoid.types.orchestrator import Mode, OrchestratorMode
from humanoid.types.robot import (
    CartesianVelocity,
    RobotBaseCommand,
    RobotConfig,
    RobotJointCommand,
    RobotName,
    RobotState,
    RobotToolCommand,
    RobotToolConfig,
)

assert TRISKEL_CONFIG.operational_space_config is not None
TRISKEL_CONTROLLER_DT = TRISKEL_CONFIG.operational_space_config.dt


def _make_robot_config() -> RobotConfig:
    return RobotConfig(
        name=RobotName.PANDA,
        tool=RobotToolConfig(frame="panda_hand"),
        homing_presets={
            HomingPreset.HOME: np.zeros(7),
            HomingPreset.REST: np.ones(7),
        },
        joint_drives={
            f"joint_{i}": JointDriveConfig(control_mode=ActuatorControlMode.POSITION)
            for i in range(7)
        },
        hardware=None,
    )


def _make_controller(
    robot_config: RobotConfig | None = None,
    *,
    clock=None,
) -> RobotControllerNode:
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
        mock_robot.hold_configuration.side_effect = lambda state, _: state.joint_positions.copy()
        mock_robot.model.nq = nq
        mock_robot.model.nv = nv
        # Geometry helpers used to initialize the hold.
        gripper_q_indices = [nq - 1] if robot_config.gripper is not None else []
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

        def compute_gripper_control(positions, *, dt=None):
            q = gripper_state["q"].copy()
            q[gripper_q_indices] = positions
            v = np.zeros(nv)
            if dt is not None:
                v[mock_gripper.controlled_v_indices] = 0.25
            return ControlResult(q=q, v=v)

        mock_gripper.update_state.side_effect = update_gripper_state
        mock_gripper.compute_control.side_effect = compute_gripper_control
        mock_gripper_cls.return_value = mock_gripper

        kwargs = {"clock": clock} if clock is not None else {}
        return RobotControllerNode(robot_config=robot_config, **kwargs)


def _no_messages(topic, timeout=0):
    return None


def _make_tool_cmd(gripper=None, velocity: CartesianVelocity | None = None):
    return RobotToolCommand(
        timestamp=0.0,
        pose=pin.SE3(np.eye(3), np.array([0.3, 0.0, 0.4])),
        gripper_positions=gripper,
        velocity=velocity,
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
    controller.current_mode = Mode.QUEST
    controller.current_tool_command = _make_tool_cmd()


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
    controller.arm_controller.compute_control.assert_not_called()
    controller.arm_controller.update_state.assert_not_called()


class TestInactiveMode:
    def test_inactive_caches_measurements_without_running_the_controller(self, controller):
        state = _make_state()
        controller.subscriber.receive.side_effect = lambda topic: (
            state if topic is Topic.ROBOT_STATE else None
        )
        controller.step()
        assert controller._last_robot_state is state
        controller.arm_controller.update_state.assert_not_called()

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
        controller.arm_controller.compute_control.assert_not_called()


class TestModeTransitions:
    def test_orchestrator_mode_message_updates_current_mode(self, controller):
        assert controller.current_mode is Mode.IDLE

        def receive(topic, timeout=0):
            if topic == Topic.ORCHESTRATOR_MODE:
                return OrchestratorMode(timestamp=0.0, mode=Mode.QUEST)
            return None

        controller.subscriber.receive = Mock(side_effect=receive)
        controller.step()

        assert controller.current_mode is Mode.QUEST
        assert controller.is_active

    def test_system_mode_activates_controller(self, controller):
        def receive(topic, timeout=0):
            if topic == Topic.ORCHESTRATOR_MODE:
                return OrchestratorMode(timestamp=0.0, mode=Mode.SYSTEM)
            return None

        controller.subscriber.receive = Mock(side_effect=receive)
        controller.step()

        assert controller.current_mode is Mode.SYSTEM
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
    def test_first_state_after_mode_entry_seeds_a_stationary_hold(self, controller):
        state = _make_state()
        messages = {Topic.ORCHESTRATOR_MODE: OrchestratorMode(timestamp=0, mode=Mode.QUEST)}
        controller.subscriber.receive.side_effect = lambda topic: messages.pop(topic, None)
        controller.step()
        controller.publisher.publish.assert_not_called()
        messages.update({Topic.ROBOT_STATE: state, Topic.ROBOT_TOOL_COMMAND: _make_tool_cmd()})
        controller.step()
        assert not messages
        command = controller.publisher.publish.call_args.args[0]
        np.testing.assert_array_equal(command.joint_positions, state.joint_positions)
        np.testing.assert_array_equal(command.joint_velocities, 0)

    def test_state_does_not_update_after_tool_command(self, active_controller):
        """Once commanded, feedback does not replace the open-loop arm state."""
        active_controller.arm_controller.configuration = MagicMock()  # already initialized
        active_controller.current_tool_command = _make_tool_cmd()
        state = _make_state()
        state.joint_positions += 100.0

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_STATE:
                return state
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()

        active_controller.arm_controller.update_state.assert_called_once()
        np.testing.assert_allclose(
            active_controller.arm_controller.update_state.call_args.args[0],
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

        active_controller.arm_controller.compute_control.assert_called_once()
        call_kwargs = active_controller.arm_controller.compute_control.call_args
        # First positional arg is the tool pose.
        assert call_kwargs.args[0] is tool_cmd.pose
        assert call_kwargs.kwargs == {"dt": pytest.approx(active_controller._nominal_dt)}

        active_controller.publisher.publish.assert_called_once()
        published = active_controller.publisher.publish.call_args
        assert isinstance(published.args[0], RobotJointCommand)
        np.testing.assert_allclose(published.args[0].joint_positions, np.arange(7, dtype=float))
        np.testing.assert_allclose(
            published.args[0].joint_velocities, np.arange(7, dtype=float) * 0.1
        )
        assert published.kwargs["topic"] is Topic.CONTROLLER_JOINT_COMMAND

    def test_base_command_on_fixed_base_keeps_arm_hold(self, active_controller):
        """A fixed-base robot continues holding its arm."""
        base_cmd = _make_base_cmd()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_BASE_COMMAND:
                return base_cmd
            return None

        active_controller.subscriber.receive = Mock(side_effect=receive)
        active_controller.step()

        # The previous tool target remains active.
        assert active_controller.current_base_command is base_cmd
        active_controller.arm_controller.compute_control.assert_called_once()
        active_controller.publisher.publish.assert_called_once()

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

        assert active_controller.arm_controller.compute_control.call_count == expected_call_count
        assert active_controller.publisher.publish.call_count == expected_call_count

    def test_velocity_feedforward_continues_reference_between_commands(self):
        now = 1.0
        controller = _make_controller(clock=lambda: now)
        _activate(controller)
        velocity = CartesianVelocity(
            linear=np.array([0.1, 0.0, 0.0]),
            angular=np.zeros(3),
        )
        tool_cmd = _make_tool_cmd(velocity=velocity)
        delivered = False

        def receive(topic, timeout=0):
            nonlocal delivered
            if topic == Topic.ROBOT_TOOL_COMMAND and not delivered:
                delivered = True
                return tool_cmd
            return None

        subscriber = cast(MagicMock, controller.subscriber)
        subscriber.receive = Mock(side_effect=receive)
        controller.step()
        now += controller._nominal_dt
        controller.step()

        arm_controller = cast(MagicMock, controller.arm_controller)
        first_call, second_call = arm_controller.compute_control.call_args_list
        np.testing.assert_allclose(first_call.args[0].translation, tool_cmd.pose.translation)
        np.testing.assert_allclose(
            second_call.args[0].translation,
            tool_cmd.pose.translation + velocity.linear * controller._nominal_dt,
        )
        for call in (first_call, second_call):
            forwarded_velocity = call.kwargs["target_velocity"]
            np.testing.assert_allclose(forwarded_velocity.linear, velocity.linear)
            np.testing.assert_allclose(forwarded_velocity.angular, velocity.angular)

    def test_base_command_alone_drives_mobile_base(self, active_mobile_controller):
        base_cmd = _make_base_cmd()

        def receive(topic, timeout=0):
            if topic == Topic.ROBOT_BASE_COMMAND:
                return base_cmd
            return None

        active_mobile_controller.subscriber.receive = Mock(side_effect=receive)
        active_mobile_controller.step()

        active_mobile_controller.base_controller.compute_control.assert_called_once_with(
            base_cmd.pose,
            dt=pytest.approx(TRISKEL_CONTROLLER_DT),
        )
        active_mobile_controller.arm_controller.compute_control.assert_called_once()
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
            tool_cmd.pose,
            dt=pytest.approx(TRISKEL_CONTROLLER_DT),
        )
        active_mobile_controller.base_controller.compute_control.assert_called_once_with(
            base_cmd.pose,
            dt=pytest.approx(TRISKEL_CONTROLLER_DT),
        )

    def test_gripper_positions_use_separate_controller(self, active_mobile_controller):
        gripper = np.array([0.01])
        tool_cmd = _make_tool_cmd(gripper=gripper)
        arm_result = active_mobile_controller.arm_controller.compute_control.return_value
        osc_gripper_positions = []

        def compute_arm_control(tool_pose, *, dt=None):
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

        active_mobile_controller.gripper_controller.compute_control.assert_called_once_with(
            gripper,
            dt=pytest.approx(TRISKEL_CONTROLLER_DT),
        )
        assert osc_gripper_positions == [pytest.approx(gripper[0])]
        arm_call = active_mobile_controller.arm_controller.compute_control.call_args
        assert arm_call.args == (tool_cmd.pose,)
        assert arm_call.kwargs == {"dt": pytest.approx(TRISKEL_CONTROLLER_DT)}
        published = active_mobile_controller.publisher.publish.call_args.args[0]
        assert published.joint_positions[-1] == pytest.approx(gripper[0])
        assert published.joint_velocities[-1] == pytest.approx(0.25)


def test_measured_control_timestep_is_used_and_bounded():
    now = 1.0
    controller = _make_controller(TRISKEL_CONFIG, clock=lambda: now)
    nominal_dt = TRISKEL_CONTROLLER_DT

    assert controller._control_timestep() == pytest.approx(nominal_dt)
    now += nominal_dt * 0.75
    assert controller._control_timestep() == pytest.approx(nominal_dt * 0.75)
    now = 1.5
    assert controller._control_timestep() == pytest.approx(nominal_dt * 2.0)
    now += nominal_dt * 0.25
    assert controller._control_timestep() == pytest.approx(nominal_dt * 0.5)


def test_close_closes_subscriber(controller):
    controller.close()
    controller.subscriber.close.assert_called_once()


@pytest.mark.parametrize("mode", [Mode.KEYBOARD, Mode.QUEST, Mode.AUTONOMY, Mode.SYSTEM])
def test_base_tracks_targets_from_its_commanded_pose(mode: Mode):
    now = 1.0
    with (
        patch("humanoid.nodes.robot.controller.Subscriber") as subscriber,
        patch("humanoid.nodes.robot.controller.Publisher") as publisher,
    ):
        controller = RobotControllerNode(TRISKEL_CONFIG, clock=lambda: now)
    controller.current_mode = mode
    home = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME]
    state = _make_state(controller.robot.model.nq, controller.robot.model.nv)
    state.joint_positions = home.copy()
    target = controller.robot.get_base_pose(home)
    assert target is not None
    target = target.copy()
    # Small enough for the controller's predicted pose to reach it in one tick.
    distance = 0.0001
    target.translation[0] += distance
    controller._seed_hold(state)
    messages = {
        Topic.ROBOT_STATE: state,
        Topic.ROBOT_BASE_COMMAND: RobotBaseCommand(timestamp=now, pose=target),
    }
    subscriber.return_value.receive.side_effect = messages.get

    for tick, measured_x in enumerate((0.0, 0.0, distance / 2, distance)):
        state.timestamp = now
        state.joint_positions[0] = measured_x
        publisher.return_value.publish.reset_mock()
        controller.step()
        publisher.return_value.publish.assert_called_once()
        command = publisher.return_value.publish.call_args.args[0]
        expected_vx = distance / TRISKEL_CONTROLLER_DT if tick == 0 else 0.0
        assert command.joint_velocities[0] == pytest.approx(expected_vx)
        messages.pop(Topic.ROBOT_BASE_COMMAND, None)
        now += TRISKEL_CONTROLLER_DT


def test_autonomy_keeps_tracking_last_targets_when_agent_stops_publishing():
    now = 1.0
    controller = _make_controller(TRISKEL_CONFIG, clock=lambda: now)
    controller.current_mode = Mode.AUTONOMY
    state = _make_state(controller.robot.model.nq, controller.robot.model.nv)
    state.timestamp = now
    tool = _make_tool_cmd(gripper=np.array([1.5]))
    base = _make_base_cmd()
    controller._seed_hold(state)
    messages = {
        Topic.ROBOT_STATE: state,
        Topic.ROBOT_TOOL_COMMAND: tool,
        Topic.ROBOT_BASE_COMMAND: base,
    }
    cast(MagicMock, controller.subscriber).receive = Mock(side_effect=messages.get)
    controller.step()
    messages.clear()
    now += 5.0
    publisher = cast(MagicMock, controller.publisher)
    publisher.publish.reset_mock()

    controller.step()

    publisher.publish.assert_called_once()
    assert cast(MagicMock, controller.arm_controller).compute_control.call_args.args == (tool.pose,)
    assert cast(MagicMock, controller.base_controller).compute_control.call_args.args == (
        base.pose,
    )
    command = publisher.publish.call_args.args[0]
    np.testing.assert_array_equal(
        command.joint_positions[controller.robot.get_gripper_position_indices()],
        tool.gripper_positions,
    )


@pytest.mark.parametrize(
    ("old_mode", "new_mode"),
    [
        (Mode.KEYBOARD, Mode.AUTONOMY),
        (Mode.AUTONOMY, Mode.QUEST),
        (Mode.KEYBOARD, Mode.QUEST),
        (Mode.IDLE, Mode.QUEST),
        (Mode.HOMING, Mode.QUEST),
        (Mode.AUTONOMY, Mode.KEYBOARD),
        (Mode.QUEST, Mode.KEYBOARD),
        (Mode.SYSTEM, Mode.KEYBOARD),
        (Mode.IDLE, Mode.KEYBOARD),
        (Mode.HOMING, Mode.KEYBOARD),
    ],
)
def test_handoff_holds_routed_joint_positions_and_stops_base(old_mode, new_mode):
    now = 1.0
    with (
        patch("humanoid.nodes.robot.controller.Subscriber") as subscriber,
        patch("humanoid.nodes.robot.controller.Publisher") as publisher,
    ):
        controller = RobotControllerNode(TRISKEL_CONFIG, clock=lambda: now)
    controller.current_mode = old_mode
    robot = controller.robot
    q = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy()
    state = _make_state(robot.model.nq, robot.model.nv)
    state.joint_positions = q.copy()
    state.timestamp = now
    state.joint_positions[:2] = [0.3, -0.2]
    indices = robot.position_controlled_q_indices
    q[indices[0]] += 0.1
    q[indices[-1]] = 1.5
    messages = {
        Topic.ROBOT_JOINT_COMMAND: RobotJointCommand(now, q, np.ones(robot.model.nv)),
        Topic.ORCHESTRATOR_MODE: OrchestratorMode(timestamp=now, mode=new_mode),
        Topic.ROBOT_STATE: state,
        Topic.ROBOT_TOOL_COMMAND: _make_tool_cmd(),
        Topic.ROBOT_BASE_COMMAND: _make_base_cmd(),
    }
    subscriber.return_value.receive.side_effect = lambda topic: messages.pop(topic, None)
    expected = state.joint_positions.copy()
    expected[indices] = q[indices]
    for _ in range(2):
        publisher.return_value.publish.reset_mock()
        controller.step()
        assert not messages
        publisher.return_value.publish.assert_called_once()
        command = publisher.return_value.publish.call_args.args[0]
        np.testing.assert_allclose(command.joint_positions, expected, atol=1e-10)
        np.testing.assert_allclose(command.joint_velocities, 0, atol=1e-10)
        now += TRISKEL_CONTROLLER_DT
