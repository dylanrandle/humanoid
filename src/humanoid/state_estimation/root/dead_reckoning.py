"""Planar dead-reckoning root-state estimator."""

import time
from collections.abc import Callable

import numpy as np

from humanoid.state_estimation.root.base import (
    PLANAR_VELOCITY_SIZE,
    RootState,
)


class DeadReckoningIntegrator:
    """Integrate the latest body-frame planar velocity into a root pose.

    The integrator is measurement-source agnostic. Concrete root-state
    estimators own the kinematics or sensors that produce its velocity input.
    """

    def __init__(
        self,
        initial_state: RootState,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._position = initial_state.position.astype(float, copy=True)
        self._velocity = initial_state.velocity.astype(float, copy=True)
        self._clock = clock
        self._last_update_time = self._clock()

    def update_velocity(self, velocity: np.ndarray) -> None:
        if velocity.shape != (PLANAR_VELOCITY_SIZE,):
            raise ValueError(
                f"Root velocity must have {PLANAR_VELOCITY_SIZE} values; received {velocity.shape}."
            )
        if not np.isfinite(velocity).all():
            raise ValueError("Root velocity values must all be finite.")
        self._integrate()
        self._velocity = velocity.astype(float, copy=True)

    def read_state(self) -> RootState:
        self._integrate()
        return RootState(position=self._position.copy(), velocity=self._velocity.copy())

    def _integrate(self) -> None:
        now = self._clock()
        dt = now - self._last_update_time
        self._last_update_time = now
        if dt <= 0.0:
            return

        yaw = float(np.arctan2(self._position[3], self._position[2]))
        vx, vy, yaw_rate = self._velocity
        midpoint_yaw = yaw + float(yaw_rate) * dt / 2
        self._position[0] += (np.cos(midpoint_yaw) * vx - np.sin(midpoint_yaw) * vy) * dt
        self._position[1] += (np.sin(midpoint_yaw) * vx + np.cos(midpoint_yaw) * vy) * dt
        yaw += float(yaw_rate) * dt
        self._position[2] = np.cos(yaw)
        self._position[3] = np.sin(yaw)
