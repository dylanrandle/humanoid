import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.robot.elrobot_mobile import ELROBOT_MOBILE_CONFIG
from humanoid.robots.base import Robot
from humanoid.robots.wheels import WheelKinematics


def _measured_velocity(robot: Robot, wheel_velocities: dict[str, float]) -> np.ndarray:
    velocity = np.zeros(robot.model.nv)
    for joint_name, joint_velocity in wheel_velocities.items():
        joint_idx = robot.joint_name_to_idx(joint_name)
        velocity[robot.joint_idx_to_velocity_idx(joint_idx)] = joint_velocity
    return velocity


@pytest.mark.parametrize(
    ("wheel_velocities", "expected_root_velocity"),
    [
        pytest.param(
            {"wheel_1": 4.0, "wheel_2": -2.0, "wheel_3": -2.0},
            [0.2, 0.0, 0.0],
            id="forward",
        ),
        pytest.param(
            {"wheel_1": 0.0, "wheel_2": 2 * np.sqrt(3), "wheel_3": -2 * np.sqrt(3)},
            [0.0, 0.2, 0.0],
            id="lateral",
        ),
        pytest.param(
            {"wheel_1": 2.0, "wheel_2": 2.0, "wheel_3": 2.0},
            [0.0, 0.0, 0.859290669],
            id="yaw",
        ),
    ],
)
def test_estimates_planar_body_velocity_from_measured_wheel_rates(
    wheel_velocities: dict[str, float],
    expected_root_velocity: list[float],
):
    robot = Robot(ELROBOT_MOBILE_CONFIG)
    kinematics = WheelKinematics(robot)
    q = pin.neutral(robot.model)
    v = _measured_velocity(robot, wheel_velocities)

    root_velocity = kinematics.estimate_root_velocity(q, v)

    np.testing.assert_allclose(root_velocity, expected_root_velocity, atol=1e-6)


def test_estimate_does_not_mutate_measured_velocity_vector():
    robot = Robot(ELROBOT_MOBILE_CONFIG)
    kinematics = WheelKinematics(robot)
    q = pin.neutral(robot.model)
    v = np.arange(robot.model.nv, dtype=float)
    original = v.copy()

    kinematics.estimate_root_velocity(q, v)

    np.testing.assert_array_equal(v, original)
