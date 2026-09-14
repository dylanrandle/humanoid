"""Cartesian figure-eight and gripper trajectory generation."""

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.robots.base import Robot
from humanoid.robots.utils.controller_tracking.models import (
    BOUNDS_EPSILON,
    FIGURE_EIGHT_LOOPS_PER_SETTING,
    PLANE_AXES,
    ControllerTrackingSettings,
    FigureEightSetting,
    GripperBounds,
)
from humanoid.types.robot import CartesianVelocity

TOOL_FORWARD_AXIS = 2


def figure_eight_pose(
    anchor_pose: pin.SE3,
    elapsed_s: float,
    settings: ControllerTrackingSettings,
    setting: FigureEightSetting,
) -> pin.SE3:
    """Return one configured figure-eight pose at ``elapsed_s``.

    The setting traces two loops with smooth acceleration and deceleration. A
    bounded rotation tilts the tool's local z axis toward the center of the path
    and smoothly returns to the anchor orientation at every center crossing.
    """
    offset, _, rotation_vector, _ = _figure_eight_kinematics(
        anchor_pose,
        elapsed_s,
        settings,
        setting,
    )
    return pin.SE3(
        pin.exp3(rotation_vector) @ anchor_pose.rotation,
        anchor_pose.translation + offset,
    )


def figure_eight_offset(
    elapsed_s: float,
    settings: ControllerTrackingSettings,
    setting: FigureEightSetting,
) -> NDArray[np.float64]:
    """Return the Cartesian offset for one figure-eight setting."""
    offset, _ = _figure_eight_translation(elapsed_s, settings, setting)
    return offset


def figure_eight_velocity(
    anchor_pose: pin.SE3,
    elapsed_s: float,
    settings: ControllerTrackingSettings,
    setting: FigureEightSetting,
) -> CartesianVelocity:
    """Return analytic linear and angular feedforward for the figure eight."""
    _, linear, rotation_vector, rotation_vector_rate = _figure_eight_kinematics(
        anchor_pose,
        elapsed_s,
        settings,
        setting,
    )
    # Pinocchio's Jexp3 maps the rotation-vector derivative to body angular
    # velocity. Its transpose gives the world/command-frame velocity used by OSC.
    angular = pin.Jexp3(rotation_vector).T @ rotation_vector_rate
    return CartesianVelocity(linear=linear, angular=angular)


def _figure_eight_kinematics(
    anchor_pose: pin.SE3,
    elapsed_s: float,
    settings: ControllerTrackingSettings,
    setting: FigureEightSetting,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    offset, offset_rate = _figure_eight_translation(elapsed_s, settings, setting)
    scale = setting.size_multiplier
    maximum_radius = 0.5 * scale * float(np.hypot(settings.width_m, settings.height_m))
    inward_fraction = -offset / maximum_radius
    inward_fraction_rate = -offset_rate / maximum_radius
    forward = anchor_pose.rotation[:, TOOL_FORWARD_AXIS]
    rotation_vector = settings.orientation_bias_rad * np.cross(forward, inward_fraction)
    rotation_vector_rate = settings.orientation_bias_rad * np.cross(
        forward,
        inward_fraction_rate,
    )
    return offset, offset_rate, rotation_vector, rotation_vector_rate


def _figure_eight_translation(
    elapsed_s: float,
    settings: ControllerTrackingSettings,
    setting: FigureEightSetting,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    phase, phase_rate = _figure_eight_phase(elapsed_s, settings)
    scale = setting.size_multiplier
    first_axis, second_axis = PLANE_AXES[setting.plane]

    offset = np.zeros(3)
    offset_rate = np.zeros(3)
    offset[first_axis] = scale * settings.width_m * 0.5 * np.sin(phase)
    offset[second_axis] = scale * settings.height_m * 0.5 * np.sin(2.0 * phase)
    offset_rate[first_axis] = scale * settings.width_m * 0.5 * phase_rate * np.cos(phase)
    offset_rate[second_axis] = scale * settings.height_m * phase_rate * np.cos(2.0 * phase)
    return offset, offset_rate


def _figure_eight_phase(
    elapsed_s: float,
    settings: ControllerTrackingSettings,
) -> tuple[float, float]:
    stage_duration_s = settings.setting_duration_s
    stage_elapsed_s = float(np.clip(elapsed_s, 0.0, stage_duration_s))
    transition_s = settings.transition_duration_s
    phase_rate_cruise = 2.0 * np.pi / settings.period_s
    phase_span = 2.0 * np.pi * FIGURE_EIGHT_LOOPS_PER_SETTING
    if stage_elapsed_s < transition_s:
        normalized_time = stage_elapsed_s / transition_s
        phase = phase_rate_cruise * transition_s * _smootherstep_integral(normalized_time)
        phase_rate = phase_rate_cruise * _smootherstep(normalized_time)
    elif stage_elapsed_s > stage_duration_s - transition_s:
        remaining_s = stage_duration_s - stage_elapsed_s
        normalized_time = remaining_s / transition_s
        phase = phase_span - (
            phase_rate_cruise * transition_s * _smootherstep_integral(normalized_time)
        )
        phase_rate = phase_rate_cruise * _smootherstep(normalized_time)
    else:
        phase = phase_rate_cruise * (stage_elapsed_s - 0.5 * transition_s)
        phase_rate = phase_rate_cruise
    return phase, phase_rate


def gripper_sinusoid(
    elapsed_s: float,
    initial_positions_rad: NDArray[np.float64],
    lower_bounds_rad: NDArray[np.float64],
    upper_bounds_rad: NDArray[np.float64],
    settings: ControllerTrackingSettings,
) -> NDArray[np.float64]:
    """Return a bounded whole-cycle sinusoid beginning and ending at rest."""
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


def _smootherstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _smootherstep_integral(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**6 - 3.0 * value**5 + 2.5 * value**4
