"""Common interface for robot control strategies."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from humanoid.types.controllers import ControlResult


class Controller[TargetT](ABC):
    """Transform a target and full-model state into a full-model command."""

    controlled_q_indices: NDArray[np.int_]
    controlled_v_indices: NDArray[np.int_]

    @abstractmethod
    def update_state(self, q: NDArray[np.float64]) -> None:
        """Replace the full-model configuration used by the controller."""

    @abstractmethod
    def compute_control(self, target: TargetT, dt: float | None = None) -> ControlResult:
        """Compute a full-model command for the requested target and elapsed timestep."""
