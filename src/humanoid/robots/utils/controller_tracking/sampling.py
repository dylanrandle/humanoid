"""Tracking sample construction and statistics."""

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.robots.base import Robot
from humanoid.robots.utils.controller_tracking.models import (
    ErrorStatistics,
    RuntimeFeedback,
    Segment,
    TrackingSample,
    TrackingStatistics,
)
from humanoid.types.robot import CartesianVelocity


def tracking_statistics(samples: list[TrackingSample]) -> TrackingStatistics:
    """Summarize tool, arm-joint, and gripper errors for non-empty samples."""
    if not samples:
        raise ValueError("Cannot summarize an empty tracking run")
    osc_position_errors = np.array(
        [np.linalg.norm(sample.osc_position_error_m) for sample in samples]
    )
    end_to_end_position_errors = np.array(
        [np.linalg.norm(sample.end_to_end_position_error_m) for sample in samples]
    )
    osc_orientation_errors = np.array([sample.osc_orientation_error_rad for sample in samples])
    end_to_end_orientation_errors = np.array(
        [sample.end_to_end_orientation_error_rad for sample in samples]
    )
    _, _, arm_joint_errors = _arm_joint_sample_matrices(samples)
    _, _, gripper_errors = _gripper_sample_matrices(samples)
    return TrackingStatistics(
        osc_position_m=_error_statistics(osc_position_errors),
        osc_orientation_rad=_error_statistics(osc_orientation_errors),
        end_to_end_position_m=_error_statistics(end_to_end_position_errors),
        end_to_end_orientation_rad=_error_statistics(end_to_end_orientation_errors),
        arm_joint_position_rad=tuple(
            _error_statistics(np.abs(arm_joint_errors[:, index]))
            for index in range(arm_joint_errors.shape[1])
        ),
        gripper_position_rad=tuple(
            _error_statistics(np.abs(gripper_errors[:, index]))
            for index in range(gripper_errors.shape[1])
        ),
    )


def _error_statistics(values: NDArray[np.float64]) -> ErrorStatistics:
    return ErrorStatistics(
        rms=float(np.sqrt(np.mean(np.square(values)))),
        mean=float(np.mean(values)),
        p95=float(np.percentile(values, 95)),
        maximum=float(np.max(values)),
    )


def _gripper_sample_matrices(
    samples: list[TrackingSample],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if not samples:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty.copy(), empty.copy()

    expected_names = samples[0].gripper_joint_names
    first_values = (
        samples[0].commanded_gripper_positions_rad,
        samples[0].measured_gripper_positions_rad,
        samples[0].gripper_position_errors_rad,
    )
    if all(values is None for values in first_values):
        if expected_names:
            raise ValueError("Gripper joint names require tracking values")
        if any(
            values is not None or sample.gripper_joint_names
            for sample in samples
            for values in (
                sample.commanded_gripper_positions_rad,
                sample.measured_gripper_positions_rad,
                sample.gripper_position_errors_rad,
            )
        ):
            raise ValueError("Gripper data must be present for every tracking sample or none")
        empty = np.empty((len(samples), 0), dtype=float)
        return empty, empty.copy(), empty.copy()
    if any(values is None for values in first_values):
        raise ValueError("Each gripper sample must contain command, measurement, and error")

    assert first_values[0] is not None
    expected_shape = (len(expected_names),)
    if first_values[0].ndim != 1:
        raise ValueError("Gripper tracking values must be one-dimensional")
    if not expected_names or first_values[0].shape != expected_shape:
        raise ValueError("Gripper tracking values must match the configured joint names")

    fields: list[list[NDArray[np.float64]]] = [[], [], []]
    for sample in samples:
        if sample.gripper_joint_names != expected_names:
            raise ValueError("Gripper joint names must be consistent across tracking samples")
        sample_values = (
            sample.commanded_gripper_positions_rad,
            sample.measured_gripper_positions_rad,
            sample.gripper_position_errors_rad,
        )
        if any(values is None or values.shape != expected_shape for values in sample_values):
            raise ValueError("Gripper tracking values must have consistent shapes")
        for field, values in zip(fields, sample_values, strict=True):
            assert values is not None
            field.append(values)
    return np.vstack(fields[0]), np.vstack(fields[1]), np.vstack(fields[2])


def _arm_joint_sample_matrices(
    samples: list[TrackingSample],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return consistent controller-command, measurement, and error matrices."""
    if not samples:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty.copy(), empty.copy()

    expected_names = samples[0].arm_joint_names
    expected_shape = (len(expected_names),)
    fields: list[list[NDArray[np.float64]]] = [[], [], []]
    for sample in samples:
        if sample.arm_joint_names != expected_names:
            raise ValueError("Arm joint names must be consistent across tracking samples")
        sample_values = (
            sample.commanded_arm_joint_positions_rad,
            sample.measured_arm_joint_positions_rad,
            sample.arm_joint_position_errors_rad,
        )
        if any(values.shape != expected_shape for values in sample_values):
            raise ValueError("Arm joint tracking values must match the configured joint names")
        for field, values in zip(fields, sample_values, strict=True):
            field.append(values)
    return np.vstack(fields[0]), np.vstack(fields[1]), np.vstack(fields[2])


def _tracking_sample(  # noqa: PLR0913 - sample fields come from distinct telemetry sources
    segment: Segment,
    setting: str,
    elapsed_s: float,
    reference_pose: pin.SE3,
    commanded_gripper_positions_rad: NDArray[np.float64] | None,
    feedback: RuntimeFeedback,
    robot: Robot,
    *,
    commanded_velocity: CartesianVelocity | None = None,
) -> TrackingSample:
    robot_state = feedback.state
    joint_command = feedback.joint_command
    osc_pose = robot.get_tool_command_pose(joint_command.joint_positions)
    measured_pose = robot.get_tool_command_pose(robot_state.joint_positions)
    osc_position_error = reference_pose.translation - osc_pose.translation
    end_to_end_position_error = reference_pose.translation - measured_pose.translation
    osc_orientation_error = pin.log3(osc_pose.rotation.T @ reference_pose.rotation)
    end_to_end_orientation_error = pin.log3(measured_pose.rotation.T @ reference_pose.rotation)
    gripper_indices = robot.get_gripper_position_indices()
    gripper_velocity_indices = robot.get_joint_velocity_indices(robot.get_gripper_joint_indices())
    gripper_joint_names = (
        robot.config.gripper.joint_names if robot.config.gripper is not None else ()
    )
    measured_gripper_positions_rad = (
        robot_state.joint_positions[gripper_indices].copy()
        if commanded_gripper_positions_rad is not None
        else None
    )
    gripper_position_errors_rad = (
        commanded_gripper_positions_rad - measured_gripper_positions_rad
        if commanded_gripper_positions_rad is not None
        and measured_gripper_positions_rad is not None
        else None
    )
    commanded_gripper_velocities_rad_s = (
        joint_command.joint_velocities[gripper_velocity_indices].copy()
        if commanded_gripper_positions_rad is not None
        and joint_command.joint_velocities is not None
        else None
    )
    measured_gripper_velocities_rad_s = (
        robot_state.joint_velocities[gripper_velocity_indices].copy()
        if commanded_gripper_positions_rad is not None
        else None
    )
    arm_joint_indices = robot.get_arm_joint_indices()
    arm_joint_velocity_indices = robot.get_joint_velocity_indices(arm_joint_indices)
    arm_joint_names = tuple(robot.joint_idx_to_name(index) for index in arm_joint_indices)
    commanded_arm_joint_positions_rad = np.array(
        [
            robot.joint_position_from_q(joint_command.joint_positions, index)
            for index in arm_joint_indices
        ]
    )
    measured_arm_joint_positions_rad = np.array(
        [
            robot.joint_position_from_q(robot_state.joint_positions, index)
            for index in arm_joint_indices
        ]
    )
    raw_arm_joint_errors = commanded_arm_joint_positions_rad - measured_arm_joint_positions_rad
    arm_joint_position_errors_rad = np.arctan2(
        np.sin(raw_arm_joint_errors), np.cos(raw_arm_joint_errors)
    )
    commanded_arm_joint_velocities_rad_s = (
        joint_command.joint_velocities[arm_joint_velocity_indices].copy()
        if joint_command.joint_velocities is not None
        else None
    )
    measured_arm_joint_velocities_rad_s = robot_state.joint_velocities[
        arm_joint_velocity_indices
    ].copy()
    return TrackingSample(
        segment=segment,
        setting=setting,
        elapsed_s=elapsed_s,
        state_timestamp_s=robot_state.timestamp,
        reference_position_m=reference_pose.translation.copy(),
        osc_position_m=osc_pose.translation.copy(),
        measured_position_m=measured_pose.translation.copy(),
        osc_position_error_m=osc_position_error,
        end_to_end_position_error_m=end_to_end_position_error,
        osc_orientation_error_rad=float(np.linalg.norm(osc_orientation_error)),
        end_to_end_orientation_error_rad=float(np.linalg.norm(end_to_end_orientation_error)),
        joint_command_timestamp_s=joint_command.timestamp,
        state_minus_joint_command_s=robot_state.timestamp - joint_command.timestamp,
        arm_joint_names=arm_joint_names,
        commanded_arm_joint_positions_rad=commanded_arm_joint_positions_rad,
        measured_arm_joint_positions_rad=measured_arm_joint_positions_rad,
        arm_joint_position_errors_rad=arm_joint_position_errors_rad,
        commanded_arm_joint_velocities_rad_s=commanded_arm_joint_velocities_rad_s,
        measured_arm_joint_velocities_rad_s=measured_arm_joint_velocities_rad_s,
        gripper_joint_names=gripper_joint_names,
        commanded_gripper_positions_rad=(
            commanded_gripper_positions_rad.copy()
            if commanded_gripper_positions_rad is not None
            else None
        ),
        measured_gripper_positions_rad=measured_gripper_positions_rad,
        gripper_position_errors_rad=gripper_position_errors_rad,
        commanded_gripper_velocities_rad_s=commanded_gripper_velocities_rad_s,
        measured_gripper_velocities_rad_s=measured_gripper_velocities_rad_s,
        commanded_linear_velocity_m_s=(
            commanded_velocity.linear.copy() if commanded_velocity is not None else None
        ),
        commanded_angular_velocity_rad_s=(
            commanded_velocity.angular.copy() if commanded_velocity is not None else None
        ),
    )
