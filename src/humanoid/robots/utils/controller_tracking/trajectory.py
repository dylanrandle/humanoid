"""Trajectory generation and endpoint validation."""

import math

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.robots.base import Robot
from humanoid.robots.utils.controller_tracking.models import (
    BOUNDS_EPSILON,
    PLANE_AXES,
    ControllerTrackingSettings,
    GripperBounds,
    JointTarget,
    ResolvedTrackingComparison,
)
from humanoid.types.actuator import ActuatorControlMode
from humanoid.types.controller_tracking import ControllerTrackingComparisonConfig
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import CartesianVelocity, RobotConfig


def figure_eight_offset(
    elapsed_s: float,
    settings: ControllerTrackingSettings,
) -> NDArray[np.float64]:
    """Return the smoothly ramped Cartesian figure-eight offset."""
    elapsed_s = float(np.clip(elapsed_s, 0.0, settings.duration_s))
    phase = 2.0 * np.pi * elapsed_s / settings.period_s
    envelope = _trajectory_envelope(elapsed_s, settings.duration_s, settings.ramp_s)
    first_axis, second_axis = PLANE_AXES[settings.plane]

    offset = np.zeros(3)
    offset[first_axis] = envelope * settings.width_m * 0.5 * np.sin(phase)
    offset[second_axis] = envelope * settings.height_m * 0.5 * np.sin(2.0 * phase)
    return offset


def figure_eight_velocity(
    elapsed_s: float,
    settings: ControllerTrackingSettings,
) -> CartesianVelocity:
    """Return the analytic command-frame velocity of the figure eight."""
    elapsed_s = float(np.clip(elapsed_s, 0.0, settings.duration_s))
    phase_rate = 2.0 * np.pi / settings.period_s
    phase = phase_rate * elapsed_s
    envelope = _trajectory_envelope(elapsed_s, settings.duration_s, settings.ramp_s)
    envelope_rate = _trajectory_envelope_rate(
        elapsed_s,
        settings.duration_s,
        settings.ramp_s,
    )
    first_axis, second_axis = PLANE_AXES[settings.plane]

    linear = np.zeros(3)
    linear[first_axis] = (
        settings.width_m
        * 0.5
        * (envelope_rate * np.sin(phase) + envelope * phase_rate * np.cos(phase))
    )
    linear[second_axis] = (
        settings.height_m
        * 0.5
        * (envelope_rate * np.sin(2.0 * phase) + 2.0 * envelope * phase_rate * np.cos(2.0 * phase))
    )
    return CartesianVelocity(linear=linear, angular=np.zeros(3))


def gripper_sinusoid(
    elapsed_s: float,
    initial_positions_rad: NDArray[np.float64],
    lower_bounds_rad: NDArray[np.float64],
    upper_bounds_rad: NDArray[np.float64],
    settings: ControllerTrackingSettings,
) -> NDArray[np.float64]:
    """Return a bounded whole-cycle sinusoid beginning and ending at rest.

    The sinusoid's phase, rather than its position, follows a minimum-jerk
    time law. This preserves the sinusoidal reversals while making position
    and velocity continuous at the measured starting position. Bounds expand
    to include that starting position so a joint already inside the URDF limit
    but outside the diagnostic margin is never commanded to jump.
    """
    elapsed_s = float(np.clip(elapsed_s, 0.0, settings.duration_s))
    if elapsed_s <= 0.0 or elapsed_s >= settings.duration_s:
        return initial_positions_rad.copy()

    effective_lower = np.minimum(lower_bounds_rad, initial_positions_rad)
    effective_upper = np.maximum(upper_bounds_rad, initial_positions_rad)
    midpoint = 0.5 * (effective_lower + effective_upper)
    amplitude = 0.5 * (effective_upper - effective_lower)
    initial_phase = np.arccos(np.clip((initial_positions_rad - midpoint) / amplitude, -1.0, 1.0))
    cycle_progress = settings.gripper_cycle_count * _smootherstep(elapsed_s / settings.duration_s)
    phase = initial_phase + 2.0 * np.pi * cycle_progress
    return midpoint + amplitude * np.cos(phase)


def resolve_tracking_comparison(
    robot: Robot,
    config: ControllerTrackingComparisonConfig,
) -> ResolvedTrackingComparison:
    """Resolve and validate corresponding joint/task endpoints for one robot."""
    expected_joint_names = {
        name
        for name, mode in robot.config.actuator_control_modes.items()
        if mode is ActuatorControlMode.POSITION
    }
    expected_frame = robot.config.base.frame if robot.config.base is not None else "world"

    resolved_joint_positions: list[NDArray[np.float64]] = []
    resolved_task_poses: list[pin.SE3] = []
    for label, endpoint in (("start", config.start), ("end", config.end)):
        provided_joint_names = set(endpoint.joint_positions_rad)
        if provided_joint_names != expected_joint_names:
            missing = sorted(expected_joint_names - provided_joint_names)
            unexpected = sorted(provided_joint_names - expected_joint_names)
            raise RuntimeError(
                f"Comparison {label} joint names do not match position-controlled joints; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if endpoint.task_frame != expected_frame:
            raise RuntimeError(
                f"Comparison {label} task pose uses frame {endpoint.task_frame!r}, "
                f"but {robot.config.name.value} tool commands use {expected_frame!r}"
            )

        joint_positions = robot.joint_positions_to_q(
            {
                robot.joint_name_to_idx(name): position
                for name, position in endpoint.joint_positions_rad.items()
            }
        )
        task_quaternion = pin.Quaternion(*endpoint.task_quaternion_wxyz)
        task_quaternion.normalize()
        task_pose = pin.SE3(
            task_quaternion.toRotationMatrix(),
            np.asarray(endpoint.task_position_m, dtype=float),
        )
        _validate_comparison_endpoint(robot, label, joint_positions, task_pose)
        resolved_joint_positions.append(joint_positions)
        resolved_task_poses.append(task_pose)

    return ResolvedTrackingComparison(
        start_joint_positions=resolved_joint_positions[0],
        end_joint_positions=resolved_joint_positions[1],
        start_task_pose=resolved_task_poses[0],
        end_task_pose=resolved_task_poses[1],
    )


def interpolated_cartesian_comparison_pose(
    start_pose: pin.SE3,
    end_pose: pin.SE3,
    elapsed_s: float,
    duration_s: float,
) -> pin.SE3:
    """Evaluate a minimum-jerk Cartesian interpolation between two poses."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("Cartesian comparison duration must be positive and finite")
    blend = _smootherstep(elapsed_s / duration_s)
    translation = start_pose.translation + blend * (end_pose.translation - start_pose.translation)
    rotation_delta = pin.log3(start_pose.rotation.T @ end_pose.rotation)
    rotation = start_pose.rotation @ pin.exp3(blend * rotation_delta)
    return pin.SE3(rotation, translation)


def interpolated_cartesian_comparison_velocity(
    start_pose: pin.SE3,
    end_pose: pin.SE3,
    elapsed_s: float,
    duration_s: float,
) -> CartesianVelocity:
    """Evaluate the command-frame velocity of the minimum-jerk interpolation."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("Cartesian comparison duration must be positive and finite")
    normalized_time = float(np.clip(elapsed_s / duration_s, 0.0, 1.0))
    blend_rate = _smootherstep_derivative(normalized_time) / duration_s
    linear = blend_rate * (end_pose.translation - start_pose.translation)
    rotation_delta = pin.log3(start_pose.rotation.T @ end_pose.rotation)
    pose = interpolated_cartesian_comparison_pose(
        start_pose,
        end_pose,
        elapsed_s,
        duration_s,
    )
    angular = pose.rotation @ (blend_rate * rotation_delta)
    return CartesianVelocity(linear=linear, angular=angular)


def _validate_comparison_endpoint(
    robot: Robot,
    label: str,
    joint_positions: NDArray[np.float64],
    task_pose: pin.SE3,
) -> None:
    if np.any(joint_positions < robot.model.lowerPositionLimit - BOUNDS_EPSILON) or np.any(
        joint_positions > robot.model.upperPositionLimit + BOUNDS_EPSILON
    ):
        raise RuntimeError(f"Comparison {label} joint pose exceeds the robot model limits")

    forward_kinematics_pose = robot.get_tool_command_pose(joint_positions)
    position_error_m = float(
        np.linalg.norm(forward_kinematics_pose.translation - task_pose.translation)
    )
    orientation_error_rad = float(
        np.linalg.norm(pin.log3(forward_kinematics_pose.rotation.T @ task_pose.rotation))
    )
    endpoint_tolerance = 1e-6
    if position_error_m > endpoint_tolerance or orientation_error_rad > endpoint_tolerance:
        raise RuntimeError(
            f"Comparison {label} joint/task poses disagree with forward kinematics: "
            f"{position_error_m:.6g} m, {orientation_error_rad:.6g} rad"
        )


def joint_space_targets(
    robot_config: RobotConfig,
    cycles: int,
) -> tuple[JointTarget, ...]:
    """Return an initial home move followed by home-to-rest-to-home round trips."""
    if cycles < 0:
        raise ValueError("joint cycles must be non-negative")
    if cycles == 0:
        return ()

    home = robot_config.homing_presets[HomingPreset.HOME]
    rest = robot_config.homing_presets[HomingPreset.REST]
    targets: list[JointTarget] = [(HomingPreset.HOME, home)]
    for _ in range(cycles):
        targets.extend(
            (
                (HomingPreset.REST, rest),
                (HomingPreset.HOME, home),
            )
        )
    return tuple(targets)


def _trajectory_envelope(elapsed_s: float, duration_s: float, ramp_s: float) -> float:
    if ramp_s == 0.0:
        return 1.0
    if elapsed_s < ramp_s:
        return _smootherstep(elapsed_s / ramp_s)
    if elapsed_s > duration_s - ramp_s:
        return _smootherstep((duration_s - elapsed_s) / ramp_s)
    return 1.0


def _trajectory_envelope_rate(elapsed_s: float, duration_s: float, ramp_s: float) -> float:
    if ramp_s == 0.0:
        return 0.0
    if elapsed_s < ramp_s:
        return _smootherstep_derivative(elapsed_s / ramp_s) / ramp_s
    if elapsed_s > duration_s - ramp_s:
        return -_smootherstep_derivative((duration_s - elapsed_s) / ramp_s) / ramp_s
    return 0.0


def _resolve_gripper_bounds(
    robot: Robot,
    settings: ControllerTrackingSettings,
) -> GripperBounds | None:
    limits = robot.get_gripper_limits()
    if not settings.move_gripper or not limits:
        return None

    model_bounds = np.asarray(limits, dtype=float)
    model_lower = model_bounds[:, 0]
    model_upper = model_bounds[:, 1]
    if not np.all(np.isfinite(model_bounds)) or np.any(model_lower >= model_upper):
        raise RuntimeError(f"The robot model has invalid gripper limits: {limits}")
    if settings.gripper_min_rad is not None:
        assert settings.gripper_max_rad is not None
        lower = np.full(len(limits), settings.gripper_min_rad)
        upper = np.full(len(limits), settings.gripper_max_rad)
        if np.any(lower < model_lower - BOUNDS_EPSILON) or np.any(
            upper > model_upper + BOUNDS_EPSILON
        ):
            raise RuntimeError(
                "Requested gripper bounds are outside the robot model limits: "
                f"requested [{settings.gripper_min_rad:g}, {settings.gripper_max_rad:g}] rad, "
                f"model {limits}"
            )
    else:
        margin = settings.gripper_limit_margin_fraction * (model_upper - model_lower)
        lower = model_lower + margin
        upper = model_upper - margin

    if np.any(lower >= upper):
        raise RuntimeError("The configured gripper limits do not contain a usable range")
    return lower, upper


def _commanded_gripper_positions(
    elapsed_s: float,
    initial_positions_rad: NDArray[np.float64] | None,
    bounds: GripperBounds | None,
    settings: ControllerTrackingSettings,
) -> NDArray[np.float64] | None:
    if initial_positions_rad is None:
        return None
    if bounds is None:
        return initial_positions_rad.copy()
    lower, upper = bounds
    return gripper_sinusoid(elapsed_s, initial_positions_rad, lower, upper, settings)


def _interpolated_gripper_positions(
    start_positions_rad: NDArray[np.float64] | None,
    end_positions_rad: NDArray[np.float64] | None,
    blend: float,
) -> NDArray[np.float64] | None:
    if start_positions_rad is None and end_positions_rad is None:
        return None
    if start_positions_rad is None or end_positions_rad is None:
        raise ValueError("Comparison gripper endpoints must both be present or absent")
    return start_positions_rad + blend * (end_positions_rad - start_positions_rad)


def _joint_command_gripper_positions(
    robot: Robot,
    joint_positions: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    gripper_indices = robot.get_gripper_position_indices()
    return joint_positions[gripper_indices].copy() if gripper_indices else None


def _smootherstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _smootherstep_derivative(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 30.0 * value**2 * (value - 1.0) ** 2
