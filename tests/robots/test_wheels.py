import numpy as np
import pinocchio as pin

from humanoid.config.robot.elrobot_mobile import ELROBOT_MOBILE_CONFIG
from humanoid.robots.base import Robot
from humanoid.robots.wheels import WheelKinematics


def test_estimates_body_velocity_from_measured_wheel_rates():
    robot = Robot(ELROBOT_MOBILE_CONFIG)
    kinematics = WheelKinematics(robot)
    q = pin.neutral(robot.model)
    v = np.zeros(robot.model.nv)
    measured_wheel_velocities = {
        "wheel_1": 4.0,
        "wheel_2": -2.0,
        "wheel_3": -2.0,
    }
    for joint_name, velocity in measured_wheel_velocities.items():
        joint_idx = robot.joint_name_to_idx(joint_name)
        v[robot.joint_idx_to_velocity_idx(joint_idx)] = velocity

    root_velocity = kinematics.estimate_root_velocity(q, v)

    np.testing.assert_allclose(root_velocity, [0.2, 0.0, 0.0], atol=1e-6)


def test_estimate_does_not_mutate_measured_velocity_vector():
    robot = Robot(ELROBOT_MOBILE_CONFIG)
    kinematics = WheelKinematics(robot)
    q = pin.neutral(robot.model)
    v = np.arange(robot.model.nv, dtype=float)
    original = v.copy()

    kinematics.estimate_root_velocity(q, v)

    np.testing.assert_array_equal(v, original)
