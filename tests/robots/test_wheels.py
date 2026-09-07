from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.robot.triskel import TRISKEL_CONFIG
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
    robot = Robot(TRISKEL_CONFIG)
    kinematics = WheelKinematics(robot)
    q = pin.neutral(robot.model)
    v = _measured_velocity(robot, wheel_velocities)

    root_velocity = kinematics.estimate_root_velocity(q, v)

    np.testing.assert_allclose(root_velocity, expected_root_velocity, atol=1e-6)


def test_estimate_does_not_mutate_measured_velocity_vector():
    robot = Robot(TRISKEL_CONFIG)
    kinematics = WheelKinematics(robot)
    q = pin.neutral(robot.model)
    v = np.arange(robot.model.nv, dtype=float)
    original = v.copy()

    kinematics.estimate_root_velocity(q, v)

    np.testing.assert_array_equal(v, original)


@pytest.mark.parametrize(
    ("root_velocity", "expected_wheel_velocities"),
    [
        pytest.param([0.2, 0.0, 0.0], [4.0, -2.0, -2.0], id="forward"),
        pytest.param(
            [0.0, 0.2, 0.0],
            [0.0, 2 * np.sqrt(3), -2 * np.sqrt(3)],
            id="lateral",
        ),
        pytest.param([0.0, 0.0, 0.859290669], [2.0, 2.0, 2.0], id="yaw"),
    ],
)
def test_computes_wheel_rates_for_planar_body_velocity(
    root_velocity: list[float],
    expected_wheel_velocities: list[float],
):
    robot = Robot(TRISKEL_CONFIG)
    kinematics = WheelKinematics(robot)
    q = pin.neutral(robot.model)

    wheel_velocity = kinematics.compute_wheel_velocities(q, np.asarray(root_velocity))

    np.testing.assert_allclose(wheel_velocity, expected_wheel_velocities, atol=3e-6)


def test_rejects_wheel_geometry_that_cannot_control_planar_root():
    wheels = TRISKEL_CONFIG.wheels
    assert wheels is not None
    config = replace(TRISKEL_CONFIG, wheels=wheels[:2])
    robot = Robot(config)
    kinematics = WheelKinematics(robot)

    with pytest.raises(RuntimeError, match="cannot control planar root velocity"):
        kinematics.compute_wheel_velocities(pin.neutral(robot.model), np.zeros(3))
