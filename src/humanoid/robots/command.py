"""Shared normalization for commands crossing a robot actuation boundary."""

import numpy as np

from humanoid.types.robot import NormalizedRobotJointCommand, RobotJointCommand


def normalize_robot_joint_command(
    command: RobotJointCommand,
    lower_position_limits: np.ndarray,
    upper_position_limits: np.ndarray,
    velocity_limits: np.ndarray,
) -> NormalizedRobotJointCommand:
    """Validate a generalized command, fill missing velocities, and clamp limits."""

    expected_positions = len(lower_position_limits)
    if command.joint_positions.shape != (expected_positions,):
        raise ValueError(
            "Robot joint command has "
            f"{len(command.joint_positions)} positions; the selected robot requires "
            f"{expected_positions}."
        )
    if not np.isfinite(command.joint_positions).all():
        raise ValueError("Robot joint command positions must all be finite.")

    expected_velocities = len(velocity_limits)
    if command.joint_velocities is not None:
        if command.joint_velocities.shape != (expected_velocities,):
            raise ValueError(
                "Robot joint command has "
                f"{len(command.joint_velocities)} velocities; the selected robot requires "
                f"{expected_velocities}."
            )
        if not np.isfinite(command.joint_velocities).all():
            raise ValueError("Robot joint command velocities must all be finite.")

    return NormalizedRobotJointCommand(
        joint_positions=np.clip(
            command.joint_positions,
            lower_position_limits,
            upper_position_limits,
        ),
        joint_velocities=(
            np.clip(command.joint_velocities, -velocity_limits, velocity_limits)
            if command.joint_velocities is not None
            else np.zeros_like(velocity_limits)
        ),
    )
