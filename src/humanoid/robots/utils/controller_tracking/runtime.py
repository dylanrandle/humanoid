"""Live controller-tracking experiment execution."""

import math
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
from humanoid.robots.utils.controller_tracking.endpoints import (
    CONTROLLER_TRACKING_COMPARISONS,
)
from humanoid.robots.utils.controller_tracking.models import (
    ControllerTrackingSettings,
    GripperBounds,
    JointTarget,
    ResolvedTrackingComparison,
    RuntimeFeedback,
    Segment,
    TrackingRun,
    TrackingSample,
)
from humanoid.robots.utils.controller_tracking.sampling import _tracking_sample
from humanoid.robots.utils.controller_tracking.timing import ControllerCommandTimingRecorder
from humanoid.robots.utils.controller_tracking.trajectory import (
    _commanded_gripper_positions,
    _interpolated_gripper_positions,
    _joint_command_gripper_positions,
    _resolve_gripper_bounds,
    _smootherstep,
    figure_eight_offset,
    figure_eight_velocity,
    interpolated_cartesian_comparison_pose,
    interpolated_cartesian_comparison_velocity,
    joint_space_targets,
    resolve_tracking_comparison,
)
from humanoid.types.controller_tracking import ControllerCommandTiming
from humanoid.types.homing import HomingPreset
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
    """Run baseline, figure-eight, and command-space comparison experiments."""
    robot = Robot(robot_config)
    comparison_config = CONTROLLER_TRACKING_COMPARISONS.get(robot_config.name)
    if comparison_config is None:
        raise RuntimeError(
            f"No controller-tracking comparison endpoints are configured for "
            f"{robot_config.name.value}"
        )
    comparison = resolve_tracking_comparison(robot, comparison_config)
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
        start = time.monotonic()

        home_target = robot_config.homing_presets[HomingPreset.HOME]
        targets = joint_space_targets(robot_config, settings.joint_cycles)
        baseline_samples = samples if targets else []
        startup_targets = targets or ((HomingPreset.HOME, home_target),)
        motion_requested = True
        feedback = _run_joint_space_trajectories(
            startup_targets,
            settings,
            robot,
            orchestrator,
            subscriber,
            feedback,
            baseline_samples,
            start,
            timing_recorder,
        )
        feedback = _wait_for_home_convergence(
            settings,
            robot,
            subscriber,
            feedback,
            home_target,
            baseline_samples,
            start,
            timing_recorder,
        )

        gripper_indices = robot.get_gripper_position_indices()
        initial_gripper_positions = (
            feedback.state.joint_positions[gripper_indices].copy() if gripper_indices else None
        )
        figure_eight_anchor = robot.get_tool_command_pose(feedback.state.joint_positions)
        _log_figure_eight_plan(figure_eight_anchor)
        feedback = _run_figure_eight(
            figure_eight_anchor,
            settings,
            robot,
            publisher,
            orchestrator,
            subscriber,
            feedback,
            samples,
            start,
            initial_gripper_positions,
            gripper_bounds,
            robot_config,
            timing_recorder,
        )

        logger.info("Moving to the supplied comparison start joint pose")
        comparison_setup_samples: list[TrackingSample] = []
        feedback = _run_homing_target(
            "comparison start",
            comparison.start_joint_positions,
            None,
            settings,
            robot,
            orchestrator,
            subscriber,
            feedback,
            comparison_setup_samples,
            start,
            timing_recorder,
        )
        feedback = _wait_for_joint_convergence(
            settings,
            robot,
            subscriber,
            feedback,
            comparison.start_joint_positions,
            comparison_setup_samples,
            start,
            segment="joint_comparison_settle",
            target_label="comparison start",
        )

        _log_comparison_plan(comparison, settings)
        feedback = _run_homing_target(
            "comparison end",
            comparison.end_joint_positions,
            "joint_comparison",
            settings,
            robot,
            orchestrator,
            subscriber,
            feedback,
            samples,
            start,
            timing_recorder,
        )
        feedback = _wait_for_joint_convergence(
            settings,
            robot,
            subscriber,
            feedback,
            comparison.end_joint_positions,
            samples,
            start,
            segment="joint_comparison_settle",
            target_label="comparison end",
            timing_recorder=timing_recorder,
        )

        logger.info("Returning to the supplied start joint pose before the Cartesian comparison")
        comparison_reset_samples: list[TrackingSample] = []
        feedback = _run_homing_target(
            "comparison start",
            comparison.start_joint_positions,
            None,
            settings,
            robot,
            orchestrator,
            subscriber,
            feedback,
            comparison_reset_samples,
            start,
            timing_recorder,
        )
        feedback = _wait_for_joint_convergence(
            settings,
            robot,
            subscriber,
            feedback,
            comparison.start_joint_positions,
            comparison_reset_samples,
            start,
            segment="cartesian_comparison_settle",
            target_label="comparison start",
        )
        feedback = _run_cartesian_comparison(
            comparison.start_task_pose,
            comparison.end_task_pose,
            comparison.start_joint_positions,
            comparison.end_joint_positions,
            settings,
            robot,
            publisher,
            orchestrator,
            subscriber,
            feedback,
            samples,
            start,
            robot_config,
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


def _run_joint_space_trajectories(  # noqa: PLR0913 - owns one diagnostic phase
    targets: tuple[JointTarget, ...],
    settings: ControllerTrackingSettings,
    robot: Robot,
    orchestrator: OrchestratorClient,
    subscriber: Subscriber,
    feedback: RuntimeFeedback,
    samples: list[TrackingSample],
    run_started_s: float,
    timing_recorder: ControllerCommandTimingRecorder,
) -> RuntimeFeedback:
    """Run named homing trajectories while recording controller output and feedback."""
    logger.info("Starting %d homing-policy transition(s)", len(targets))
    for transition_index, (preset, target) in enumerate(targets, start=1):
        segment: Segment = "joint_home" if preset is HomingPreset.HOME else "joint_rest"
        feedback = _run_homing_target(
            f"{preset.value} ({transition_index}/{len(targets)})",
            target,
            segment,
            settings,
            robot,
            orchestrator,
            subscriber,
            feedback,
            samples,
            run_started_s,
            timing_recorder,
        )
    return feedback


def _run_homing_target(  # noqa: PLR0913 - owns one homing transition
    label: str,
    target: NDArray[np.float64],
    segment: Segment | None,
    settings: ControllerTrackingSettings,
    robot: Robot,
    orchestrator: OrchestratorClient,
    subscriber: Subscriber,
    feedback: RuntimeFeedback,
    samples: list[TrackingSample],
    run_started_s: float,
    timing_recorder: ControllerCommandTimingRecorder | None = None,
) -> RuntimeFeedback:
    """Run one target through the homing controller, optionally recording it."""
    logger.info("Joint-space homing transition: %s", label)
    if segment is not None and timing_recorder is not None:
        timing_recorder.begin(segment)
    requested_at = time.perf_counter()
    orchestrator.request_homing(target)
    _wait_for_mode(subscriber, Mode.HOMING, settings.connection_timeout_s)
    joint_command = _wait_for_fresh_joint_command(
        subscriber,
        minimum_timestamp_s=requested_at,
        timeout_s=settings.connection_timeout_s,
    )
    received_at = time.monotonic()
    feedback = replace(
        feedback,
        joint_command=joint_command,
        mode=Mode.HOMING,
        last_joint_command_received_s=received_at,
        last_mode_received_s=received_at,
    )
    next_tick = received_at
    allowed_modes = frozenset({Mode.HOMING, Mode.IDLE})

    while True:
        feedback = _refresh_runtime_state(
            subscriber,
            feedback,
            settings.feedback_timeout_s,
            allowed_modes=allowed_modes,
        )
        if segment is not None:
            command_pose = robot.get_tool_command_pose(feedback.joint_command.joint_positions)
            command_gripper_positions = _joint_command_gripper_positions(
                robot,
                feedback.joint_command.joint_positions,
            )
            samples.append(
                _tracking_sample(
                    segment,
                    time.monotonic() - run_started_s,
                    command_pose,
                    command_gripper_positions,
                    feedback,
                    robot,
                )
            )
        if feedback.mode is Mode.IDLE:
            break
        next_tick = _sleep_until_next_tick(next_tick, settings.rate_hz)
    if segment is not None and timing_recorder is not None:
        timing_recorder.end()
    return feedback


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
    robot_config: RobotConfig,
    timing_recorder: ControllerCommandTimingRecorder,
) -> RuntimeFeedback:
    """Run the Cartesian figure eight through OSC/IK."""
    logger.info(
        "Running %.0f x %.0f mm Cartesian figure eight through OSC/IK over %.1f seconds",
        settings.width_m * 1_000.0,
        settings.height_m * 1_000.0,
        settings.duration_s,
    )
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
            f"but {robot_config.name.value} expects {robot.model.nq}; check --robot"
        )
    feedback_received_at = time.monotonic()
    feedback = replace(
        feedback,
        joint_command=initial_joint_command,
        mode=Mode.SYSTEM,
        last_joint_command_received_s=feedback_received_at,
        last_mode_received_s=feedback_received_at,
    )

    timing_recorder.begin("figure_eight")
    trajectory_started_s = time.monotonic()
    next_tick = trajectory_started_s
    while True:
        trajectory_elapsed_s = min(
            time.monotonic() - trajectory_started_s,
            settings.duration_s,
        )
        feedback = _refresh_runtime_state(subscriber, feedback, settings.feedback_timeout_s)
        command = RobotToolCommand(
            timestamp=time.perf_counter(),
            pose=pin.SE3(
                anchor_pose.rotation.copy(),
                anchor_pose.translation + figure_eight_offset(trajectory_elapsed_s, settings),
            ),
            gripper_positions=_commanded_gripper_positions(
                trajectory_elapsed_s,
                initial_gripper_positions,
                gripper_bounds,
                settings,
            ),
            velocity=(
                figure_eight_velocity(trajectory_elapsed_s, settings)
                if settings.velocity_feedforward
                else None
            ),
        )
        publisher.publish(command, topic=Topic.SYSTEM_TOOL_COMMAND)
        samples.append(
            _tracking_sample(
                "figure_eight",
                time.monotonic() - run_started_s,
                command.pose,
                command.gripper_positions,
                feedback,
                robot,
                commanded_velocity=command.velocity,
            )
        )
        if trajectory_elapsed_s >= settings.duration_s:
            break
        next_tick = _sleep_until_next_tick(next_tick, settings.rate_hz)

    timing_recorder.end()
    timing_recorder.begin("figure_eight_settle")
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


def _run_cartesian_comparison(  # noqa: PLR0913 - owns one diagnostic phase
    start_pose: pin.SE3,
    end_pose: pin.SE3,
    start_joint_positions: NDArray[np.float64],
    end_joint_positions: NDArray[np.float64],
    settings: ControllerTrackingSettings,
    robot: Robot,
    publisher: Publisher,
    orchestrator: OrchestratorClient,
    subscriber: Subscriber,
    feedback: RuntimeFeedback,
    samples: list[TrackingSample],
    run_started_s: float,
    robot_config: RobotConfig,
    timing_recorder: ControllerCommandTimingRecorder,
) -> RuntimeFeedback:
    """Follow one smooth Cartesian start-to-end path through OSC/IK."""
    logger.info(
        "Running the Cartesian comparison through OSC/IK over %.1f seconds",
        settings.comparison_duration_s,
    )
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
            f"but {robot_config.name.value} expects {robot.model.nq}; check --robot"
        )
    feedback_received_at = time.monotonic()
    feedback = replace(
        feedback,
        joint_command=initial_joint_command,
        mode=Mode.SYSTEM,
        last_joint_command_received_s=feedback_received_at,
        last_mode_received_s=feedback_received_at,
    )

    timing_recorder.begin("cartesian_comparison")
    trajectory_started_s = time.monotonic()
    next_tick = trajectory_started_s
    start_gripper_positions = _joint_command_gripper_positions(robot, start_joint_positions)
    end_gripper_positions = _joint_command_gripper_positions(robot, end_joint_positions)
    while True:
        trajectory_elapsed_s = min(
            time.monotonic() - trajectory_started_s,
            settings.comparison_duration_s,
        )
        feedback = _refresh_runtime_state(subscriber, feedback, settings.feedback_timeout_s)
        blend = _smootherstep(trajectory_elapsed_s / settings.comparison_duration_s)
        commanded_gripper_positions = _interpolated_gripper_positions(
            start_gripper_positions,
            end_gripper_positions,
            blend,
        )
        command = RobotToolCommand(
            timestamp=time.perf_counter(),
            pose=interpolated_cartesian_comparison_pose(
                start_pose,
                end_pose,
                trajectory_elapsed_s,
                settings.comparison_duration_s,
            ),
            gripper_positions=commanded_gripper_positions,
            velocity=(
                interpolated_cartesian_comparison_velocity(
                    start_pose,
                    end_pose,
                    trajectory_elapsed_s,
                    settings.comparison_duration_s,
                )
                if settings.velocity_feedforward
                else None
            ),
        )
        publisher.publish(command, topic=Topic.SYSTEM_TOOL_COMMAND)
        samples.append(
            _tracking_sample(
                "cartesian_comparison",
                time.monotonic() - run_started_s,
                command.pose,
                command.gripper_positions,
                feedback,
                robot,
                commanded_velocity=command.velocity,
            )
        )
        if trajectory_elapsed_s >= settings.comparison_duration_s:
            break
        next_tick = _sleep_until_next_tick(next_tick, settings.rate_hz)

    timing_recorder.end()
    timing_recorder.begin("cartesian_comparison_settle")
    settle_end_s = time.monotonic() + settings.settle_s
    while time.monotonic() < settle_end_s:
        feedback = _refresh_runtime_state(subscriber, feedback, settings.feedback_timeout_s)
        command = RobotToolCommand(
            timestamp=time.perf_counter(),
            pose=end_pose,
            gripper_positions=end_gripper_positions,
            velocity=CartesianVelocity.zero() if settings.velocity_feedforward else None,
        )
        publisher.publish(command, topic=Topic.SYSTEM_TOOL_COMMAND)
        samples.append(
            _tracking_sample(
                "cartesian_comparison_settle",
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


def _wait_for_home_convergence(  # noqa: PLR0913 - records the final settling phase
    settings: ControllerTrackingSettings,
    robot: Robot,
    subscriber: Subscriber,
    feedback: RuntimeFeedback,
    home_target: NDArray[np.float64],
    samples: list[TrackingSample],
    run_started_s: float,
    timing_recorder: ControllerCommandTimingRecorder | None = None,
) -> RuntimeFeedback:
    """Wait until measured arm joints remain close to HOME."""
    return _wait_for_joint_convergence(
        settings,
        robot,
        subscriber,
        feedback,
        home_target,
        samples,
        run_started_s,
        segment="joint_home_settle",
        target_label="HOME",
        timing_recorder=timing_recorder,
    )


def _wait_for_joint_convergence(  # noqa: PLR0913 - records target settling
    settings: ControllerTrackingSettings,
    robot: Robot,
    subscriber: Subscriber,
    feedback: RuntimeFeedback,
    target: NDArray[np.float64],
    samples: list[TrackingSample],
    run_started_s: float,
    *,
    segment: Segment,
    target_label: str,
    timing_recorder: ControllerCommandTimingRecorder | None = None,
) -> RuntimeFeedback:
    """Wait until measured arm joints remain close to a requested target."""
    arm_joint_indices = robot.get_arm_joint_indices()
    if not arm_joint_indices:
        return feedback
    if timing_recorder is not None:
        timing_recorder.begin(segment)

    logger.info(
        "Waiting for measured %s convergence: tolerance %.3f rad for %.1f seconds",
        target_label,
        settings.home_position_tolerance_rad,
        settings.home_stable_s,
    )
    deadline = time.monotonic() + settings.home_timeout_s
    stable_since: float | None = None
    maximum_error_rad = math.inf
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        prior_state_timestamp = feedback.state.timestamp
        feedback = _refresh_runtime_state(
            subscriber,
            feedback,
            settings.feedback_timeout_s,
            allowed_modes=frozenset({Mode.IDLE}),
            require_fresh_joint_command=False,
        )
        if feedback.state.timestamp <= prior_state_timestamp:
            next_tick = _sleep_until_next_tick(next_tick, settings.rate_hz)
            continue

        command_pose = robot.get_tool_command_pose(feedback.joint_command.joint_positions)
        command_gripper_positions = _joint_command_gripper_positions(
            robot,
            feedback.joint_command.joint_positions,
        )
        samples.append(
            _tracking_sample(
                segment,
                time.monotonic() - run_started_s,
                command_pose,
                command_gripper_positions,
                feedback,
                robot,
            )
        )

        errors = []
        for joint_index in arm_joint_indices:
            target_position = robot.joint_position_from_q(target, joint_index)
            measured_position = robot.joint_position_from_q(
                feedback.state.joint_positions,
                joint_index,
            )
            raw_error = target_position - measured_position
            errors.append(abs(math.atan2(math.sin(raw_error), math.cos(raw_error))))
        maximum_error_rad = max(errors)
        now = time.monotonic()
        if maximum_error_rad <= settings.home_position_tolerance_rad:
            stable_since = now if stable_since is None else stable_since
            if now - stable_since >= settings.home_stable_s:
                logger.info("Measured %s target is stable", target_label)
                if timing_recorder is not None:
                    timing_recorder.end()
                return feedback
        else:
            stable_since = None
        next_tick = _sleep_until_next_tick(next_tick, settings.rate_hz)

    if timing_recorder is not None:
        timing_recorder.end()
    raise RuntimeError(
        f"Timed out waiting for measured {target_label} convergence; maximum arm-joint "
        f"error was {maximum_error_rad:.3f} rad after {settings.home_timeout_s:g} seconds"
    )


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
    *,
    allowed_modes: frozenset[Mode] = frozenset({Mode.SYSTEM}),
    require_fresh_joint_command: bool = True,
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
        if mode_message.mode not in allowed_modes:
            expected = ", ".join(sorted(mode.value for mode in allowed_modes))
            raise RuntimeError(
                f"Orchestrator entered {mode_message.mode.value}; expected {expected}; "
                "stopping test"
            )
        latest_mode = mode_message.mode
        last_mode_received = now

    if now - last_state_received > timeout_s:
        raise RuntimeError(f"Robot-state feedback was stale for more than {timeout_s:g} seconds")
    if require_fresh_joint_command and now - last_joint_command_received > timeout_s:
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
        "Controller tracking: robot=%s, plane=%s, figure_eight=%.0f x %.0f mm, "
        "period=%.1f s, cycles=%d, rate=%.1f Hz, velocity_feedforward=%s",
        robot_config.name,
        settings.plane,
        settings.width_m * 1_000.0,
        settings.height_m * 1_000.0,
        settings.period_s,
        settings.cycles,
        settings.rate_hz,
        "on" if settings.velocity_feedforward else "off",
    )
    if settings.joint_cycles:
        logger.info(
            "Motion sequence: move home, complete %d home-to-rest-to-home round trips, "
            "run the Cartesian figure eight through OSC/IK, then compare the supplied "
            "start/end poses through the homing controller and Cartesian OSC/IK",
            settings.joint_cycles,
        )
    else:
        logger.info(
            "Motion sequence: move home without recording the baseline, run the Cartesian "
            "figure eight through OSC/IK, then compare the supplied start/end poses "
            "through the homing controller and Cartesian OSC/IK"
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


def _log_comparison_plan(
    comparison: ResolvedTrackingComparison,
    settings: ControllerTrackingSettings,
) -> None:
    """Log the explicitly configured comparison trajectory."""
    logger.info(
        "Comparison Cartesian interpolation duration: %.1f seconds",
        settings.comparison_duration_s,
    )
    for label, pose in (
        ("Start", comparison.start_task_pose),
        ("End", comparison.end_task_pose),
    ):
        logger.info(
            "%s tool position: %s m",
            label,
            np.array2string(pose.translation, precision=4),
        )
