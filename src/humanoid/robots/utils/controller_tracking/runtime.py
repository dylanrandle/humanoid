"""Live Cartesian figure-eight controller-tracking experiment execution."""

import time
from dataclasses import replace

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.robots.base import Robot
from humanoid.robots.utils.controller_tracking.models import (
    FIGURE_EIGHT_LOOPS_PER_SETTING,
    FIGURE_EIGHT_PLANES,
    FIGURE_EIGHT_SETTINGS,
    FIGURE_EIGHT_SIZE_MULTIPLIERS,
    ControllerTrackingSettings,
    GripperBounds,
    RuntimeFeedback,
    TrackingRun,
    TrackingSample,
)
from humanoid.robots.utils.controller_tracking.sampling import _tracking_sample
from humanoid.robots.utils.controller_tracking.timing import ControllerCommandTimingRecorder
from humanoid.robots.utils.controller_tracking.trajectory import (
    _commanded_gripper_positions,
    _resolve_gripper_bounds,
    figure_eight_pose,
    figure_eight_velocity,
)
from humanoid.types.controller_tracking import ControllerCommandTiming
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import (
    CartesianVelocity,
    RobotConfig,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

logger = get_logger(__name__)


def run_controller_tracking(  # noqa: PLR0915 - owns the utility's safety lifecycle
    settings: ControllerTrackingSettings,
    robot_config: RobotConfig = ROBOT_CONFIG,
) -> TrackingRun:
    """Run the Cartesian figure-eight tracking experiment from the current pose."""
    robot = Robot(robot_config)
    publisher = Publisher()
    subscriber = Subscriber(
        topics=[Topic.ROBOT_STATE, Topic.ROBOT_JOINT_COMMAND, Topic.ORCHESTRATOR_MODE]
    )
    timing_recorder = ControllerCommandTimingRecorder(robot=robot)
    orchestrator = OrchestratorClient(publisher=publisher)
    motion_requested = False
    samples: list[TrackingSample] = []
    completed = False
    failure_reason: str | None = None
    controller_command_timings: list[ControllerCommandTiming] = []
    native_joint_samples = []

    try:
        initial_mode = subscriber.receive(
            Topic.ORCHESTRATOR_MODE,
            timeout=round(settings.connection_timeout_s * 1_000.0),
        )
        if initial_mode is None:
            raise RuntimeError("Timed out waiting for the orchestrator; start the main stack first")
        if initial_mode.mode is not Mode.IDLE:
            raise RuntimeError(
                f"The orchestrator must be idle before this test; current mode is "
                f"{initial_mode.mode.value}"
            )

        initial_state = subscriber.receive(
            Topic.ROBOT_STATE,
            timeout=round(settings.connection_timeout_s * 1_000.0),
        )
        if initial_state is None:
            raise RuntimeError("Timed out waiting for robot state; start the main stack first")
        if initial_state.joint_positions.shape != (robot.model.nq,):
            raise RuntimeError(
                f"Robot state has {len(initial_state.joint_positions)} positions, but "
                f"{robot_config.name.value} expects {robot.model.nq}; check --robot"
            )

        gripper_bounds = _resolve_gripper_bounds(robot, settings)
        _log_run_plan(settings, robot_config, gripper_bounds)
        _count_down(settings.start_delay_s)
        _wait_for_mode(subscriber, Mode.IDLE, min(settings.connection_timeout_s, 1.0))
        initial_state = _wait_for_fresh_robot_state(
            subscriber,
            minimum_timestamp_s=initial_state.timestamp,
            timeout_s=settings.connection_timeout_s,
        )
        feedback_received_at = time.monotonic()
        feedback = RuntimeFeedback(
            state=initial_state,
            joint_command=RobotJointCommand(
                timestamp=initial_state.timestamp,
                joint_positions=initial_state.joint_positions.copy(),
            ),
            mode=Mode.IDLE,
            last_state_received_s=feedback_received_at,
            last_joint_command_received_s=feedback_received_at,
            last_mode_received_s=feedback_received_at,
        )

        gripper_indices = robot.get_gripper_position_indices()
        initial_gripper_positions = (
            initial_state.joint_positions[gripper_indices].copy() if gripper_indices else None
        )
        anchor_pose = robot.get_tool_command_pose(initial_state.joint_positions)
        _log_figure_eight_plan(anchor_pose)
        motion_requested = True
        _run_figure_eight(
            anchor_pose,
            settings,
            robot,
            publisher,
            orchestrator,
            subscriber,
            feedback,
            samples,
            time.monotonic(),
            initial_gripper_positions,
            gripper_bounds,
            timing_recorder,
        )
        completed = True
    except KeyboardInterrupt:
        logger.warning("Controller tracking run interrupted; returning the controller to idle")
    except RuntimeError as exc:
        if not samples:
            raise
        failure_reason = str(exc)
        logger.error(
            "Controller tracking stopped early; preserving the partial run: %s",
            failure_reason,
        )
    finally:
        if motion_requested:
            orchestrator.request_idle()
        controller_command_timings = timing_recorder.close()
        native_joint_samples = timing_recorder.native_joint_samples
        subscriber.close()

    return TrackingRun(
        samples=samples,
        completed=completed,
        failure_reason=failure_reason,
        controller_command_timings=controller_command_timings,
        native_joint_samples=native_joint_samples,
    )


def _run_figure_eight(  # noqa: PLR0913 - owns one diagnostic phase
    anchor_pose: pin.SE3,
    settings: ControllerTrackingSettings,
    robot: Robot,
    publisher: Publisher,
    orchestrator: OrchestratorClient,
    subscriber: Subscriber,
    feedback: RuntimeFeedback,
    samples: list[TrackingSample],
    run_started_s: float,
    initial_gripper_positions: NDArray[np.float64] | None,
    gripper_bounds: GripperBounds | None,
    timing_recorder: ControllerCommandTimingRecorder,
) -> RuntimeFeedback:
    """Run the plane/size matrix of orientation-aware figure eights through OSC/IK."""
    initial_command = RobotToolCommand(
        timestamp=time.perf_counter(),
        pose=anchor_pose,
        gripper_positions=initial_gripper_positions,
        velocity=CartesianVelocity.zero() if settings.velocity_feedforward else None,
    )
    # Seed the system source while Idle so OSC has a current, stationary target as
    # soon as the orchestrator switches ownership to this diagnostic.
    publisher.publish(initial_command, topic=Topic.SYSTEM_TOOL_COMMAND)
    system_requested_at = time.perf_counter()
    orchestrator.request_system()
    _wait_for_mode(subscriber, Mode.SYSTEM, settings.connection_timeout_s)
    initial_joint_command = _wait_for_fresh_joint_command(
        subscriber,
        minimum_timestamp_s=system_requested_at,
        timeout_s=settings.connection_timeout_s,
    )
    if initial_joint_command.joint_positions.shape != (robot.model.nq,):
        raise RuntimeError(
            f"Controller command has {len(initial_joint_command.joint_positions)} positions, "
            f"but {robot.config.name.value} expects {robot.model.nq}; check --robot"
        )
    feedback_received_at = time.monotonic()
    feedback = replace(
        feedback,
        joint_command=initial_joint_command,
        mode=Mode.SYSTEM,
        last_joint_command_received_s=feedback_received_at,
        last_mode_received_s=feedback_received_at,
    )

    trajectory_started_s = time.monotonic()
    for setting_index, setting in enumerate(FIGURE_EIGHT_SETTINGS, start=1):
        logger.info(
            "Running figure-eight setting %d/%d: plane=%s, size=%.1fx",
            setting_index,
            len(FIGURE_EIGHT_SETTINGS),
            setting.plane,
            setting.size_multiplier,
        )
        timing_recorder.begin("figure_eight", setting.name)
        setting_started_s = time.monotonic()
        next_tick = setting_started_s
        while True:
            setting_elapsed_s = min(
                time.monotonic() - setting_started_s,
                settings.setting_duration_s,
            )
            run_trajectory_elapsed_s = min(
                time.monotonic() - trajectory_started_s,
                settings.duration_s,
            )
            feedback = _refresh_runtime_state(subscriber, feedback, settings.feedback_timeout_s)
            command = RobotToolCommand(
                timestamp=time.perf_counter(),
                pose=figure_eight_pose(anchor_pose, setting_elapsed_s, settings, setting),
                gripper_positions=_commanded_gripper_positions(
                    run_trajectory_elapsed_s,
                    initial_gripper_positions,
                    gripper_bounds,
                    settings,
                ),
                velocity=(
                    figure_eight_velocity(anchor_pose, setting_elapsed_s, settings, setting)
                    if settings.velocity_feedforward
                    else None
                ),
            )
            publisher.publish(command, topic=Topic.SYSTEM_TOOL_COMMAND)
            samples.append(
                _tracking_sample(
                    "figure_eight",
                    setting.name,
                    time.monotonic() - run_started_s,
                    command.pose,
                    command.gripper_positions,
                    feedback,
                    robot,
                    commanded_velocity=command.velocity,
                )
            )
            if setting_elapsed_s >= settings.setting_duration_s:
                break
            next_tick = _sleep_until_next_tick(next_tick, settings.rate_hz)
        timing_recorder.end()

    final_setting = FIGURE_EIGHT_SETTINGS[-1]
    timing_recorder.begin("figure_eight_settle", final_setting.name)
    settle_end_s = time.monotonic() + settings.settle_s
    while time.monotonic() < settle_end_s:
        feedback = _refresh_runtime_state(subscriber, feedback, settings.feedback_timeout_s)
        command = RobotToolCommand(
            timestamp=time.perf_counter(),
            pose=anchor_pose,
            gripper_positions=initial_gripper_positions,
            velocity=CartesianVelocity.zero() if settings.velocity_feedforward else None,
        )
        publisher.publish(command, topic=Topic.SYSTEM_TOOL_COMMAND)
        samples.append(
            _tracking_sample(
                "figure_eight_settle",
                final_setting.name,
                time.monotonic() - run_started_s,
                command.pose,
                command.gripper_positions,
                feedback,
                robot,
                commanded_velocity=command.velocity,
            )
        )
        next_tick = _sleep_until_next_tick(next_tick, settings.rate_hz)
    timing_recorder.end()
    return feedback


def _wait_for_mode(subscriber: Subscriber, expected: Mode, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_mode: Mode | None = None
    while time.monotonic() < deadline:
        remaining_ms = max(1, round((deadline - time.monotonic()) * 1_000.0))
        message = subscriber.receive(Topic.ORCHESTRATOR_MODE, timeout=min(remaining_ms, 100))
        if message is None:
            continue
        last_mode = message.mode
        if last_mode is expected:
            return
    observed = "no mode" if last_mode is None else last_mode.value
    raise RuntimeError(
        f"Timed out waiting for orchestrator mode {expected.value}; last observed {observed}"
    )


def _wait_for_fresh_joint_command(
    subscriber: Subscriber,
    minimum_timestamp_s: float,
    timeout_s: float,
) -> RobotJointCommand:
    """Wait for a controller output created after the requested mode transition."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining_ms = max(1, round((deadline - time.monotonic()) * 1_000.0))
        command = subscriber.receive(Topic.ROBOT_JOINT_COMMAND, timeout=min(remaining_ms, 100))
        if command is not None and command.timestamp >= minimum_timestamp_s:
            return command
    raise RuntimeError("Timed out waiting for a fresh controller joint command")


def _wait_for_fresh_robot_state(
    subscriber: Subscriber,
    minimum_timestamp_s: float,
    timeout_s: float,
) -> RobotState:
    """Wait for robot feedback newer than the previously observed state."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining_ms = max(1, round((deadline - time.monotonic()) * 1_000.0))
        state = subscriber.receive(Topic.ROBOT_STATE, timeout=min(remaining_ms, 100))
        if state is not None and state.timestamp > minimum_timestamp_s:
            return state
    raise RuntimeError("Timed out waiting for fresh robot-state feedback")


def _refresh_runtime_state(
    subscriber: Subscriber,
    feedback: RuntimeFeedback,
    timeout_s: float,
) -> RuntimeFeedback:
    now = time.monotonic()
    latest_state = feedback.state
    latest_joint_command = feedback.joint_command
    last_state_received = feedback.last_state_received_s
    last_joint_command_received = feedback.last_joint_command_received_s
    last_mode_received = feedback.last_mode_received_s
    latest_mode = feedback.mode
    state = subscriber.receive(Topic.ROBOT_STATE)
    if state is not None:
        latest_state = state
        last_state_received = now

    joint_command = subscriber.receive(Topic.ROBOT_JOINT_COMMAND)
    if joint_command is not None:
        latest_joint_command = joint_command
        last_joint_command_received = now

    mode_message = subscriber.receive(Topic.ORCHESTRATOR_MODE)
    if mode_message is not None:
        if mode_message.mode is not Mode.SYSTEM:
            raise RuntimeError(
                f"Orchestrator entered {mode_message.mode.value}; expected system; stopping test"
            )
        latest_mode = mode_message.mode
        last_mode_received = now

    if now - last_state_received > timeout_s:
        raise RuntimeError(f"Robot-state feedback was stale for more than {timeout_s:g} seconds")
    if now - last_joint_command_received > timeout_s:
        raise RuntimeError(
            f"Controller joint commands were stale for more than {timeout_s:g} seconds"
        )
    if now - last_mode_received > timeout_s:
        raise RuntimeError(f"Orchestrator mode was stale for more than {timeout_s:g} seconds")
    return RuntimeFeedback(
        state=latest_state,
        joint_command=latest_joint_command,
        mode=latest_mode,
        last_state_received_s=last_state_received,
        last_joint_command_received_s=last_joint_command_received,
        last_mode_received_s=last_mode_received,
    )


def _sleep_until(deadline_s: float) -> None:
    delay_s = deadline_s - time.monotonic()
    if delay_s > 0.0:
        time.sleep(delay_s)


def _sleep_until_next_tick(previous_tick: float, rate_hz: float) -> float:
    period_s = 1.0 / rate_hz
    next_tick = previous_tick + period_s
    now = time.monotonic()
    if next_tick < now:
        next_tick = now
    else:
        _sleep_until(next_tick)
    return next_tick


def _count_down(delay_s: float) -> None:
    if delay_s == 0.0:
        return
    logger.warning("Motion begins in %.1f seconds; press Ctrl-C to cancel", delay_s)
    deadline = time.monotonic() + delay_s
    while (remaining := deadline - time.monotonic()) > 0.0:
        time.sleep(min(remaining, 0.1))


def _log_run_plan(
    settings: ControllerTrackingSettings,
    robot_config: RobotConfig,
    gripper_bounds_rad: GripperBounds | None,
) -> None:
    logger.info(
        "Controller tracking: robot=%s, planes=%s, base_size=%.0f x %.0f mm, "
        "loops_per_setting=%d, size_multipliers=%s, cruise_period=%.1f s, rate=%.1f Hz, "
        "orientation_bias=%.1f deg, velocity_feedforward=%s",
        robot_config.name,
        FIGURE_EIGHT_PLANES,
        settings.width_m * 1_000.0,
        settings.height_m * 1_000.0,
        FIGURE_EIGHT_LOOPS_PER_SETTING,
        FIGURE_EIGHT_SIZE_MULTIPLIERS,
        settings.period_s,
        settings.rate_hz,
        np.rad2deg(settings.orientation_bias_rad),
        "on" if settings.velocity_feedforward else "off",
    )
    logger.info(
        "Motion matrix: two loops for every plane/size combination, ordered by increasing size"
    )
    if gripper_bounds_rad is not None:
        lower, upper = gripper_bounds_rad
        logger.info(
            "Gripper sinusoid: lower=%s rad, upper=%s rad, requested_period=%.1f s, "
            "effective_period=%.1f s, cycles=%d",
            np.array2string(lower, precision=4),
            np.array2string(upper, precision=4),
            settings.gripper_period_s,
            settings.effective_gripper_period_s,
            settings.gripper_cycle_count,
        )


def _log_figure_eight_plan(anchor_pose: pin.SE3) -> None:
    logger.info(
        "Figure-eight tool anchor: %s m",
        np.array2string(anchor_pose.translation, precision=4),
    )
