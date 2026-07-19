"""Root-state estimator construction from estimation configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from humanoid.state_estimation.root.base import RootState, RootStateEstimator
from humanoid.state_estimation.root.config import RootStateEstimatorConfig
from humanoid.state_estimation.root.wheel_dead_reckoning import (
    WheelDeadReckoningRootStateEstimator,
    WheelDeadReckoningRootStateEstimatorConfig,
)

if TYPE_CHECKING:
    from humanoid.robots.base import Robot


def create_root_state_estimator(
    config: RootStateEstimatorConfig,
    robot: Robot,
    initial_state: RootState,
) -> RootStateEstimator:
    """Create the configured root-state estimator."""
    if isinstance(config, WheelDeadReckoningRootStateEstimatorConfig):
        return WheelDeadReckoningRootStateEstimator(robot, initial_state)
    raise TypeError(f"Unsupported root-state estimator: {type(config).__name__}")
