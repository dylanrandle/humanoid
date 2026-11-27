from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.robot import SO101_CONFIG, TRISKEL_CONFIG
from humanoid.policy.motion import MotionPolicy
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset
from humanoid.types.motion import MotionTarget
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import CartesianVelocityLimits, RobotJointCommand, RobotState


@pytest.fixture
def control(monkeypatch):
    clock = Mock(return_value=10.0)
    monkeypatch.setattr("humanoid.policy.motion.time.perf_counter", clock)
    robot = Robot(TRISKEL_CONFIG)
    state = RobotState(
        timestamp=10.0,
        joint_positions=TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy(),
        joint_velocities=np.zeros(robot.model.nv),
        actuator_temperatures=np.zeros(robot.model.nq),
    )
    return MotionPolicy(robot), Observation(robot_state=state, mode=Mode.AUTONOMY), clock


@pytest.mark.parametrize(
    "command",
    [
        MotionTarget(base_delta=np.array([1.0, 0, 0])),
        MotionTarget(base_delta=np.array([0.0, 0, float("nan")])),
        MotionTarget(base_delta=np.array(["0", 0, 0])),
        MotionTarget(base_delta=np.array([True, False, False])),
        MotionTarget(base_delta=np.array([0.0, 0])),
        MotionTarget(tool_pose=pin.SE3(np.eye(3), np.array([0.0, 0, 1]))),
        MotionTarget(tool_pose=pin.SE3(np.zeros((3, 3)), np.zeros(3))),
        MotionTarget(gripper_positions=np.array([100.0])),
        MotionTarget(duration=100),
        MotionTarget(duration=True),
    ],
)
def test_rejected_command_leaves_policy_idle(control, command):
    policy, obs, _ = control
    policy(obs)
    result = policy.submit(command)
    assert not result.done()
    assert policy(obs).base_pose is None
    with pytest.raises(ValueError):
        result.result()


def test_motion_advances_only_on_policy_steps_and_reports_measured_failure(control):
    policy, obs, clock = control
    result = policy.submit(MotionTarget(duration=0.1, base_delta=np.array([0.04, 0, 0])))
    action = policy(obs)
    assert action.base_pose is not None and action.tool_pose is not None
    assert action.joint_positions is None
    assert not result.done()
    clock.return_value = 10.12
    hold = policy(obs)
    assert result.done() and not result.result()["reached"]
    expected = policy.robot.get_base_pose(obs.robot_state.joint_positions)
    np.testing.assert_allclose(hold.base_pose.homogeneous, expected.homogeneous)
    assert policy(obs).base_pose is None  # Does not keep publishing between agent commands.


@pytest.mark.parametrize("immediate_request", [False, True])
def test_autonomy_takes_over_arm_and_squeeze_from_routed_command(control, immediate_request):
    policy, obs, _ = control
    robot = policy.robot
    commanded = obs.robot_state.joint_positions.copy()
    commanded[:2] += 0.1
    commanded[robot.position_controlled_q_indices[0]] += 0.15
    grip = robot.get_gripper_position_indices()
    commanded[grip] = 1.3
    obs.robot_joint_command = RobotJointCommand(9.0, commanded)
    if immediate_request:
        policy.submit(MotionTarget(duration=1.0))
    action = policy(obs)
    np.testing.assert_allclose(
        action.tool_pose.homogeneous, robot.get_tool_command_pose(commanded).homogeneous, atol=1e-12
    )
    np.testing.assert_allclose(action.gripper_positions, commanded[grip])
    np.testing.assert_allclose(
        action.base_pose.homogeneous,
        robot.get_base_pose(obs.robot_state.joint_positions).homogeneous,
        atol=1e-12,
    )


def test_mode_change_interrupts_and_requires_new_command(control):
    policy, obs, _ = control
    result = policy.submit(MotionTarget(base_delta=np.array([0.04, 0, 0])))
    obs.mode = Mode.IDLE
    assert policy(obs).base_pose is None
    with pytest.raises(RuntimeError, match="interrupted"):
        result.result()
    obs.mode = Mode.AUTONOMY
    policy(obs)  # Re-anchor on activation.
    assert policy(obs).base_pose is None


def test_stale_feedback_interrupts_without_action(control):
    policy, obs, clock = control
    result = policy.submit(MotionTarget(base_delta=np.array([0.04, 0, 0])))
    assert policy(obs).base_pose is not None
    clock.return_value = 11.0
    interrupted = policy(obs)
    assert interrupted.base_pose is None and interrupted.tool_pose is None
    with pytest.raises(RuntimeError, match="stale"):
        result.result()


def test_cancel_holds_measured_pose_and_rejects_overlap(control):
    policy, obs, _ = control
    result = policy.submit(MotionTarget(base_delta=np.array([0.04, 0, 0])))
    with pytest.raises(RuntimeError, match="already running"):
        policy.submit(MotionTarget())
    policy.reset()
    with pytest.raises(RuntimeError, match="interrupted"):
        result.result()
    assert policy(obs).tool_pose is not None


@pytest.mark.parametrize("mode", [Mode.IDLE, None])
def test_commands_require_explicit_autonomy_mode(control, mode):
    policy, obs, _ = control
    obs.mode = mode
    result = policy.submit(MotionTarget())
    assert policy(obs).base_pose is None
    with pytest.raises(RuntimeError, match=r"UI|interrupted"):
        result.result()


def test_grip_preload_survives_completion_idle_and_arm_motion(control):
    policy, obs, clock = control
    indices = policy.robot.get_gripper_position_indices()
    obs.robot_state.joint_positions[indices] = 0.6
    result = policy.submit(MotionTarget(duration=1.0, gripper_positions=np.array([1.2])))
    policy(obs)
    clock.return_value = obs.robot_state.timestamp = 11.0
    completed = policy(obs)
    assert not result.result()["reached"]  # Contact prevents reaching the target.
    np.testing.assert_allclose(completed.gripper_positions, [1.2])
    held_tool = completed.tool_pose.homogeneous.copy()

    clock.return_value = obs.robot_state.timestamp = 12.0
    idle = policy(obs)
    np.testing.assert_allclose(idle.gripper_positions, [1.2])
    np.testing.assert_allclose(idle.tool_pose.homogeneous, held_tool)
    assert idle.base_pose is None

    tool = policy.robot.get_tool_command_pose(obs.robot_state.joint_positions)
    policy.submit(MotionTarget(tool_pose=tool))
    np.testing.assert_allclose(policy(obs).gripper_positions, [1.2])


def test_short_gripper_command_holds_only_the_rate_limited_target(control):
    policy, obs, clock = control
    indices = policy.robot.get_gripper_position_indices()
    obs.robot_state.joint_positions[indices] = 0.0
    policy.submit(MotionTarget(duration=0.1, gripper_positions=np.array([2.2])))
    policy(obs)
    clock.return_value = obs.robot_state.timestamp = 10.2
    expected = policy.gripper_speed * 0.1
    np.testing.assert_allclose(policy(obs).gripper_positions, expected)
    clock.return_value = obs.robot_state.timestamp = 11.0
    np.testing.assert_allclose(policy(obs).gripper_positions, expected)


@pytest.mark.parametrize("interruption", ["cancel", "stale", "mode"])
def test_interruption_discards_grip_preload(control, interruption):
    policy, obs, clock = control
    obs.robot_state.joint_positions[policy.robot.get_gripper_position_indices()] = 0.6
    policy.submit(MotionTarget(duration=1.0, gripper_positions=np.array([1.2])))
    policy(obs)
    clock.return_value = obs.robot_state.timestamp = 11.0
    policy(obs)
    if interruption == "cancel":
        policy.reset()
    elif interruption == "stale":
        clock.return_value = 12.0
        assert policy(obs).tool_pose is None
    else:
        obs.mode = Mode.IDLE
        assert policy(obs).tool_pose is None
        obs.mode = Mode.AUTONOMY
    obs.robot_state.timestamp = clock.return_value
    np.testing.assert_allclose(policy(obs).gripper_positions, [0.6])


def test_tool_interpolation_uses_selected_robot_velocity_limits(control):
    policy, obs, clock = control
    velocity = CartesianVelocityLimits(linear=0.01, angular=0.05)
    config = replace(
        policy.robot.config, tool=replace(policy.robot.config.tool, velocity_limits=velocity)
    )
    policy = MotionPolicy(Robot(config), config=policy.config)
    initial = policy.robot.get_tool_command_pose(obs.robot_state.joint_positions).copy()
    target = pin.SE3(initial.rotation @ pin.exp3(np.array([0.2, 0, 0])), initial.translation)
    target.translation[0] += 0.06
    policy.submit(MotionTarget(duration=0.1, tool_pose=target))
    policy(obs)
    elapsed = 0.05
    clock.return_value = obs.robot_state.timestamp = 10.0 + elapsed
    action = policy(obs)
    assert action.tool_pose is not None
    assert np.linalg.norm(action.tool_pose.translation - initial.translation) == pytest.approx(
        elapsed * velocity.linear
    )
    angle = np.linalg.norm(pin.log3(initial.rotation.T @ action.tool_pose.rotation))
    assert angle < elapsed * velocity.angular


def test_base_target_uses_selected_robot_velocity_limits(control):
    policy, obs, _ = control
    velocity = CartesianVelocityLimits(linear=0.02, angular=0.04)
    config = replace(
        policy.robot.config, base=replace(policy.robot.config.base, velocity_limits=velocity)
    )
    policy = MotionPolicy(Robot(config), config=policy.config)
    policy.submit(MotionTarget(base_delta=np.array([0.1, 0, 0.3])))
    measured = policy.robot.get_base_pose(obs.robot_state.joint_positions)
    assert measured is not None
    delta = measured.inverse() * policy(obs).base_pose
    controller = policy.robot.config.omniwheel_base_config
    assert controller is not None
    assert np.linalg.norm(delta.translation) == pytest.approx(
        velocity.linear * controller.dt / controller.position_gain
    )
    assert np.linalg.norm(pin.log3(delta.rotation)) == pytest.approx(
        velocity.angular * controller.dt / controller.orientation_gain
    )


@pytest.mark.parametrize("cancel", [False, True])
def test_agent_corrects_base_slip_in_body_frame_and_stops_at_commanded_pose(control, cancel):
    policy, obs, clock = control
    policy(obs)
    # Odometry faces +Y while the integrated command faces +X.
    measured = obs.robot_state.joint_positions
    measured[:4] = [0, 0, 0, 1]
    commanded = measured.copy()
    commanded[:4] = [0.2, -0.1, 1, 0]
    obs.robot_joint_command = RobotJointCommand(clock(), commanded)
    reference = policy.robot.get_base_pose(commanded).copy()
    # A small forward step and turn are below the per-tick limits.
    step = 0.0001
    result = policy.submit(MotionTarget(duration=0.1, base_delta=np.array([step, 0, step])))
    expected = pin.SE3(pin.rpy.rpyToMatrix(0, 0, step), np.array([step, 0, 0]))
    delta = reference.actInv(policy(obs).base_pose)
    np.testing.assert_allclose(delta.homogeneous, expected.homogeneous, atol=1e-12)

    measured[1] += step / 2
    delta = reference.actInv(policy(obs).base_pose)
    np.testing.assert_allclose(delta.translation, [step / 2, 0, 0], atol=1e-12)
    if cancel:
        policy.cancel()
    else:
        clock.return_value = obs.robot_state.timestamp = 10.11
    hold = policy(obs)
    assert result.done()
    np.testing.assert_allclose(hold.base_pose.homogeneous, reference.homogeneous, atol=1e-12)


def test_submission_from_another_thread_uses_next_observation_and_copies_arguments(control):
    policy, obs, clock = control
    delta = np.array([0.04, 0, 0])
    command = MotionTarget(base_delta=delta, duration=0.1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(policy.submit, command).result(timeout=1)
    delta[0] = 100
    assert not result.running() and not result.done()
    clock.return_value = obs.robot_state.timestamp = 11.0
    policy(obs)
    assert result.running()  # Execution begins now, not when submit() was called.
    clock.return_value = obs.robot_state.timestamp = 11.2
    policy(obs)
    outcome = result.result()
    assert outcome["position_error_m"] == pytest.approx(0.04)
    assert outcome["elapsed_seconds"] == pytest.approx(0.2)


def test_command_expiring_before_next_policy_tick_never_starts(control):
    policy, obs, clock = control
    policy(obs)
    result = policy.submit(MotionTarget(base_delta=np.array([0.04, 0, 0])), deadline=10.1)
    clock.return_value = obs.robot_state.timestamp = 10.2
    assert policy(obs).base_pose is None
    with pytest.raises(RuntimeError, match="expired"):
        result.result()


def test_cancelled_queued_future_never_starts_motion(control):
    policy, obs, _ = control
    policy(obs)
    result = policy.submit(MotionTarget(base_delta=np.array([0.04, 0, 0])))
    assert result.cancel()
    assert policy(obs).base_pose is None
    assert result.cancelled()


def test_reset_interrupts_active_command_and_allows_another_submission(control):
    policy, obs, _ = control
    result = policy.submit(MotionTarget())
    policy(obs)
    assert result.running()
    policy.reset()
    with pytest.raises(RuntimeError, match="interrupted"):
        result.result()
    next_result = policy.submit(MotionTarget())
    assert not next_result.done()


def test_simultaneous_submissions_accept_only_one_command(control):
    policy, _, _ = control
    with ThreadPoolExecutor(max_workers=2) as executor:
        submissions = [executor.submit(policy.submit, MotionTarget()) for _ in range(2)]
        errors = [submission.exception(timeout=1) for submission in submissions]
    assert sum(error is None for error in errors) == 1
    assert sum(isinstance(error, RuntimeError) for error in errors) == 1
    policy.reset()


@pytest.mark.parametrize("invalid_target", ["translation", "rotation", "nan"])
def test_tool_limits_are_enforced_for_typed_commands(control, invalid_target):
    policy, obs, _ = control
    pose = policy.robot.get_tool_command_pose(obs.robot_state.joint_positions).copy()
    if invalid_target == "translation":
        pose.translation[0] += policy.config.max_tool_translation * 1.1
    elif invalid_target == "rotation":
        pose.rotation = pose.rotation @ pin.exp3(
            np.array([policy.config.max_tool_rotation * 1.1, 0, 0])
        )
    else:
        pose.translation[0] = float("nan")
    result = policy.submit(MotionTarget(tool_pose=pose))
    policy(obs)
    with pytest.raises(ValueError):
        result.result(timeout=0)


def test_cancel_stops_motion_without_releasing_squeeze(control):
    policy, obs, clock = control
    indices = policy.robot.get_gripper_position_indices()
    obs.robot_state.joint_positions[indices] = 0.6
    result = policy.submit(MotionTarget(duration=1.0, gripper_positions=np.array([1.2])))
    policy(obs)
    clock.return_value = obs.robot_state.timestamp = 10.5
    squeeze = policy(obs).gripper_positions.copy()
    policy.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        result.result(timeout=0)
    hold = policy(obs)
    np.testing.assert_allclose(hold.gripper_positions, squeeze)
    np.testing.assert_allclose(
        hold.base_pose.homogeneous,
        policy.robot.get_base_pose(obs.robot_state.joint_positions).homogeneous,
    )


def test_invalid_deadline_does_not_queue_a_command(control):
    policy, obs, _ = control
    with pytest.raises(ValueError, match="finite"):
        policy.submit(MotionTarget(), deadline=float("nan"))
    future = policy.submit(MotionTarget())
    policy(obs)
    assert future.running()


def test_fixed_arm_controls_gripper_and_rejects_base_motion(monkeypatch):
    monkeypatch.setattr("humanoid.policy.motion.time.perf_counter", lambda: 10.0)
    robot = Robot(SO101_CONFIG)
    obs = Observation(
        mode=Mode.AUTONOMY,
        robot_state=RobotState(
            timestamp=10.0,
            joint_positions=SO101_CONFIG.homing_presets[HomingPreset.HOME].copy(),
            joint_velocities=np.zeros(robot.model.nv),
            actuator_temperatures=np.zeros(robot.model.nq),
        ),
    )
    policy = MotionPolicy(robot)
    result = policy.submit(MotionTarget(base_delta=np.zeros(3)))
    assert policy(obs).base_pose is None
    with pytest.raises(ValueError, match="mobile robot"):
        result.result()
    positions = obs.robot_state.joint_positions[robot.get_gripper_position_indices()]
    result = policy.submit(MotionTarget(duration=0.1, gripper_positions=positions))
    action = policy(obs)
    assert action.base_pose is None and action.tool_pose is not None
    np.testing.assert_array_equal(action.gripper_positions, positions)
    monkeypatch.setattr("humanoid.policy.motion.time.perf_counter", lambda: 10.2)
    policy(obs)
    assert result.result(timeout=0)["reached"]
