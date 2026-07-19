"""Wheel-feedback root-state estimation with planar dead reckoning."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from humanoid.robots.wheels import WheelKinematics
from humanoid.state_estimation.root.base import RootState, RootStateEstimator
from humanoid.state_estimation.root.config import RootStateEstimatorConfig
from humanoid.state_estimation.root.dead_reckoning import DeadReckoningIntegrator

if TYPE_CHECKING:
    from humanoid.robots.base import Robot


@dataclass(frozen=True, kw_only=True)
class WheelDeadReckoningRootStateEstimatorConfig(RootStateEstimatorConfig):
    """Configuration for root-state estimation from measured wheel feedback."""


class WheelDeadReckoningRootStateEstimator(RootStateEstimator):
    """Estimate planar root state from wheel kinematics and dead reckoning."""

    def __init__(
        self,
        robot: Robot,
        initial_state: RootState,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ):
        root_q_slice = robot.get_root_q_slice()
        if root_q_slice is None:
            raise ValueError("Wheel dead reckoning requires a planar root joint.")
        self._root_q_slice = root_q_slice
        self._wheel_kinematics = WheelKinematics(robot)
        self._dead_reckoning = DeadReckoningIntegrator(initial_state, clock=clock)

    def update(self, q: np.ndarray, v: np.ndarray) -> RootState:
        measured_configuration = q.copy()
        measured_configuration[self._root_q_slice] = self._dead_reckoning.read_state().position
        root_velocity = self._wheel_kinematics.estimate_root_velocity(
            measured_configuration,
            v,
        )
        self._dead_reckoning.update_velocity(root_velocity)
        return self._dead_reckoning.read_state()
