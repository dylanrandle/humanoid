import numpy as np
import pinocchio as pin

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


def test_composes_wheel_kinematics_and_dead_reckoning():
    robot = Robot(ELROBOT_MOBILE_CONFIG)
    clock = FakeClock()
    estimator = WheelDeadReckoningRootStateEstimator(
        robot,
        RootState(position=np.array([0.0, 0.0, 1.0, 0.0]), velocity=np.zeros(3)),
        clock=clock,
    )
    q = pin.neutral(robot.model)
    v = np.zeros(robot.model.nv)
    for joint_name, velocity in {
        "wheel_1": 4.0,
        "wheel_2": -2.0,
        "wheel_3": -2.0,
    }.items():
        joint_idx = robot.joint_name_to_idx(joint_name)
        v[robot.joint_idx_to_velocity_idx(joint_idx)] = velocity

    initial_state = estimator.update(q, v)
    clock.advance(2.0)
    updated_state = estimator.update(q, v)

    np.testing.assert_allclose(initial_state.velocity, [0.2, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(updated_state.position, [0.4, 0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(updated_state.velocity, [0.2, 0.0, 0.0], atol=1e-6)
