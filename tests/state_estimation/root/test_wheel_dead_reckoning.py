import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.robot.elrobot_mobile import ELROBOT_MOBILE_CONFIG
from humanoid.robots.base import Robot
from humanoid.state_estimation.root.base import RootState
from humanoid.state_estimation.root.wheel_dead_reckoning import (
    WheelDeadReckoningRootStateEstimator,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _measured_velocity(robot: Robot, wheel_velocities: dict[str, float]) -> np.ndarray:
    velocity = np.zeros(robot.model.nv)
    for joint_name, joint_velocity in wheel_velocities.items():
        joint_idx = robot.joint_name_to_idx(joint_name)
        velocity[robot.joint_idx_to_velocity_idx(joint_idx)] = joint_velocity
    return velocity


YAW_RATE = 0.859290669


@pytest.mark.parametrize(
    ("wheel_velocities", "expected_velocity", "expected_position"),
    [
        pytest.param(
            {"wheel_1": 4.0, "wheel_2": -2.0, "wheel_3": -2.0},
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 1.0, 0.0],
            id="forward",
        ),
        pytest.param(
            {"wheel_1": 0.0, "wheel_2": 2 * np.sqrt(3), "wheel_3": -2 * np.sqrt(3)},
            [0.0, 0.2, 0.0],
            [0.0, 0.4, 1.0, 0.0],
            id="lateral",
        ),
        pytest.param(
            {"wheel_1": 2.0, "wheel_2": 2.0, "wheel_3": 2.0},
            [0.0, 0.0, YAW_RATE],
            [0.0, 0.0, np.cos(2 * YAW_RATE), np.sin(2 * YAW_RATE)],
            id="yaw",
        ),
    ],
)
def test_composes_wheel_kinematics_and_dead_reckoning_for_planar_motion(
    wheel_velocities: dict[str, float],
    expected_velocity: list[float],
    expected_position: list[float],
):
    robot = Robot(ELROBOT_MOBILE_CONFIG)
    clock = FakeClock()
    estimator = WheelDeadReckoningRootStateEstimator(
        robot,
        RootState(position=np.array([0.0, 0.0, 1.0, 0.0]), velocity=np.zeros(3)),
        clock=clock,
    )
    q = pin.neutral(robot.model)
    v = _measured_velocity(robot, wheel_velocities)

    initial_state = estimator.update(q, v)
    clock.advance(2.0)
    updated_state = estimator.update(q, v)

    np.testing.assert_allclose(initial_state.velocity, expected_velocity, atol=1e-6)
    np.testing.assert_allclose(updated_state.position, expected_position, atol=1e-6)
    np.testing.assert_allclose(updated_state.velocity, expected_velocity, atol=1e-6)
