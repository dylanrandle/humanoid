"""Runtime-independent planar root-state interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

PLANAR_POSITION_SIZE = 4
PLANAR_VELOCITY_SIZE = 3


@dataclass(frozen=True)
class RootState:
    """Planar root position and velocity in Pinocchio coordinates.

    Position is ``[x, y, cos(yaw), sin(yaw)]`` and velocity is the body-frame
    planar twist ``[vx, vy, yaw_rate]``.
    """

    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        if self.position.shape != (PLANAR_POSITION_SIZE,):
            raise ValueError(
                f"Root position must have {PLANAR_POSITION_SIZE} values; "
                f"received {self.position.shape}."
            )
        if self.velocity.shape != (PLANAR_VELOCITY_SIZE,):
            raise ValueError(
                f"Root velocity must have {PLANAR_VELOCITY_SIZE} values; "
                f"received {self.velocity.shape}."
            )
        if not np.isfinite(self.position).all() or not np.isfinite(self.velocity).all():
            raise ValueError("Root state values must all be finite.")


class RootStateEstimator(ABC):
    """Estimate planar root state from measured robot configuration and velocity."""

    @abstractmethod
    def update(self, q: np.ndarray, v: np.ndarray) -> RootState:
        """Update the estimate from measured robot state and return the result."""
