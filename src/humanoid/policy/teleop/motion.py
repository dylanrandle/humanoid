"""Stateful Cartesian command shaping for teleoperation policies."""

import math

import numpy as np
import pinocchio as pin

from humanoid.types.robot import CartesianVelocityLimits

ZERO_DISTANCE_TOLERANCE = 1e-12


class CartesianPoseLimiter:
    """Generate velocity- and acceleration-bounded poses toward a target."""

    def __init__(
        self,
        velocity_limits: CartesianVelocityLimits,
        linear_acceleration_limit: float | None,
        angular_acceleration_limit: float | None,
    ) -> None:
        self.velocity_limits = velocity_limits
        self.linear_acceleration_limit = linear_acceleration_limit
        self.angular_acceleration_limit = angular_acceleration_limit
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def reset(self) -> None:
        """Forget velocity history, as when a dead-man switch disengages."""
        self.linear_velocity.fill(0.0)
        self.angular_velocity.fill(0.0)

    def step(self, current: pin.SE3, target: pin.SE3, dt: float) -> pin.SE3:
        """Advance ``current`` toward ``target`` within configured limits."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Cartesian limiter timestep must be positive and finite.")

        translation_error = target.translation - current.translation
        self.linear_velocity = _next_velocity(
            translation_error,
            self.linear_velocity,
            self.velocity_limits.linear,
            self.linear_acceleration_limit,
            dt,
        )
        translation_step = self.linear_velocity * dt
        translation, self.linear_velocity = _apply_without_overshoot(
            current.translation,
            target.translation,
            translation_step,
            self.linear_velocity,
        )

        rotation_error = pin.log3(current.rotation.T @ target.rotation)
        self.angular_velocity = _next_velocity(
            rotation_error,
            self.angular_velocity,
            self.velocity_limits.angular,
            self.angular_acceleration_limit,
            dt,
        )
        rotation_step = self.angular_velocity * dt
        if _reaches_target(rotation_error, rotation_step):
            rotation = target.rotation.copy()
            self.angular_velocity.fill(0.0)
        else:
            rotation = current.rotation @ pin.exp3(rotation_step)

        return pin.SE3(rotation, translation)


def low_pass_pose(previous: pin.SE3, sample: pin.SE3, dt: float, time_constant: float) -> pin.SE3:
    """Apply first-order low-pass filtering to translation and SO(3) rotation."""
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Pose-filter timestep must be positive and finite.")
    if not math.isfinite(time_constant) or time_constant < 0.0:
        raise ValueError("Pose-filter time constant must be finite and non-negative.")
    if time_constant <= 0.0:
        return sample.copy()
    alpha = 1.0 - math.exp(-dt / time_constant)
    translation = previous.translation + alpha * (sample.translation - previous.translation)
    rotation_error = pin.log3(previous.rotation.T @ sample.rotation)
    rotation = previous.rotation @ pin.exp3(alpha * rotation_error)
    return pin.SE3(rotation, translation)


def _next_velocity(
    error: np.ndarray,
    previous_velocity: np.ndarray,
    velocity_limit: float,
    acceleration_limit: float | None,
    dt: float,
) -> np.ndarray:
    distance = float(np.linalg.norm(error))
    if distance <= ZERO_DISTANCE_TOLERANCE:
        desired_velocity = np.zeros(3)
    else:
        desired_speed = min(velocity_limit, distance / dt)
        if acceleration_limit is not None:
            desired_speed = min(desired_speed, math.sqrt(2.0 * acceleration_limit * distance))
        desired_velocity = error * (desired_speed / distance)

    if acceleration_limit is None:
        return desired_velocity
    velocity_delta = desired_velocity - previous_velocity
    delta_norm = float(np.linalg.norm(velocity_delta))
    max_delta = acceleration_limit * dt
    if delta_norm > max_delta:
        velocity_delta *= max_delta / delta_norm
    return previous_velocity + velocity_delta


def _reaches_target(error: np.ndarray, step: np.ndarray) -> bool:
    error_squared = float(np.dot(error, error))
    if error_squared <= ZERO_DISTANCE_TOLERANCE**2:
        return False

    projected_scale = float(np.dot(error, step)) / error_squared
    if projected_scale < 1.0:
        return False

    lateral_step = step - projected_scale * error
    return bool(np.linalg.norm(lateral_step) <= ZERO_DISTANCE_TOLERANCE)


def _apply_without_overshoot(
    current: np.ndarray,
    target: np.ndarray,
    step: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error = target - current
    if _reaches_target(error, step):
        return target.copy(), np.zeros(3)
    return current + step, velocity
