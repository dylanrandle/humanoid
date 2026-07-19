"""Tests for robot command validation and limit normalization."""

import numpy as np
import pytest

from humanoid.robots.command import normalize_robot_joint_command
from humanoid.types.robot import RobotJointCommand


def test_normalizes_positions_and_velocities_against_model_limits():
    normalized = normalize_robot_joint_command(
        RobotJointCommand(
            timestamp=0.0,
            joint_positions=np.array([-2.0, 0.5, 2.0]),
            joint_velocities=np.array([-3.0, 0.5, 3.0]),
        ),
        lower_position_limits=np.full(3, -1.0),
        upper_position_limits=np.full(3, 1.0),
        velocity_limits=np.full(3, 2.0),
    )

    assert normalized.joint_positions == pytest.approx([-1.0, 0.5, 1.0])
    assert normalized.joint_velocities == pytest.approx([-2.0, 0.5, 2.0])


def test_normalizes_missing_velocities_to_zero():
    normalized = normalize_robot_joint_command(
        RobotJointCommand(timestamp=0.0, joint_positions=np.zeros(2)),
        lower_position_limits=np.full(2, -1.0),
        upper_position_limits=np.full(2, 1.0),
        velocity_limits=np.full(2, 2.0),
    )

    assert normalized.joint_velocities == pytest.approx(np.zeros(2))


@pytest.mark.parametrize(
    "command",
    [
        RobotJointCommand(timestamp=0.0, joint_positions=np.zeros(1)),
        RobotJointCommand(
            timestamp=0.0,
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(1),
        ),
        RobotJointCommand(timestamp=0.0, joint_positions=np.array([np.nan, 0.0])),
        RobotJointCommand(
            timestamp=0.0,
            joint_positions=np.zeros(2),
            joint_velocities=np.array([0.0, np.inf]),
        ),
    ],
)
def test_rejects_incompatible_or_non_finite_commands(command: RobotJointCommand):
    with pytest.raises(ValueError):
        normalize_robot_joint_command(
            command,
            lower_position_limits=np.full(2, -1.0),
            upper_position_limits=np.full(2, 1.0),
            velocity_limits=np.full(2, 2.0),
        )
