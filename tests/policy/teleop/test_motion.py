import math

import numpy as np
import pinocchio as pin
import pytest

from humanoid.policy.teleop.motion import CartesianPoseLimiter, low_pass_pose
from humanoid.types.robot import CartesianVelocityLimits


def _limiter(
    *,
    linear_velocity: float = 1.0,
    angular_velocity: float = 2.0,
    linear_acceleration: float | None = 0.5,
    angular_acceleration: float | None = 1.0,
) -> CartesianPoseLimiter:
    return CartesianPoseLimiter(
        CartesianVelocityLimits(linear=linear_velocity, angular=angular_velocity),
        linear_acceleration,
        angular_acceleration,
    )


def test_translation_acceleration_is_bounded_across_ticks():
    limiter = _limiter()
    current = pin.SE3.Identity()
    target = pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))

    first = limiter.step(current, target, 0.1)
    second = limiter.step(first, target, 0.1)

    np.testing.assert_allclose(first.translation, [0.005, 0.0, 0.0])
    np.testing.assert_allclose(second.translation, [0.015, 0.0, 0.0])
    np.testing.assert_allclose(limiter.linear_velocity, [0.1, 0.0, 0.0])


def test_rotation_acceleration_is_bounded_across_ticks():
    limiter = _limiter()
    current = pin.SE3.Identity()
    target = pin.SE3(pin.utils.rotate("z", 1.0), np.zeros(3))

    first = limiter.step(current, target, 0.1)
    second = limiter.step(first, target, 0.1)

    assert pin.log3(first.rotation)[2] == pytest.approx(0.01)
    assert pin.log3(second.rotation)[2] == pytest.approx(0.03)
    np.testing.assert_allclose(limiter.angular_velocity, [0.0, 0.0, 0.2], atol=1e-12)


def test_motion_does_not_overshoot_a_nearby_target():
    limiter = _limiter(linear_acceleration=None, angular_acceleration=None)
    current = pin.SE3.Identity()
    target = pin.SE3(np.eye(3), np.array([0.001, 0.0, 0.0]))

    result = limiter.step(current, target, 0.1)

    np.testing.assert_allclose(result.translation, target.translation)
    np.testing.assert_array_equal(limiter.linear_velocity, np.zeros(3))


def test_target_plane_crossing_does_not_discard_lateral_velocity():
    limiter = _limiter(linear_velocity=2.0, linear_acceleration=1.0)
    limiter.linear_velocity = np.array([1.0, 1.0, 0.0])
    previous_velocity = limiter.linear_velocity.copy()
    current = pin.SE3.Identity()
    target = pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]))

    result = limiter.step(current, target, 0.1)

    assert not np.allclose(result.translation, target.translation)
    assert np.linalg.norm(limiter.linear_velocity - previous_velocity) <= 0.1 + 1e-12
    assert limiter.linear_velocity[1] > 0.0


def test_pose_low_pass_filters_translation_and_rotation():
    previous = pin.SE3.Identity()
    sample = pin.SE3(pin.utils.rotate("z", 1.0), np.array([1.0, 0.0, 0.0]))

    result = low_pass_pose(previous, sample, dt=0.1, time_constant=0.1)

    alpha = 1.0 - math.exp(-1.0)
    np.testing.assert_allclose(result.translation, [alpha, 0.0, 0.0])
    assert pin.log3(result.rotation)[2] == pytest.approx(alpha)


def test_zero_filter_time_constant_returns_sample():
    sample = pin.SE3(pin.utils.rotate("x", 0.2), np.array([1.0, 2.0, 3.0]))

    result = low_pass_pose(pin.SE3.Identity(), sample, dt=0.1, time_constant=0.0)

    np.testing.assert_allclose(result.homogeneous, sample.homogeneous)
