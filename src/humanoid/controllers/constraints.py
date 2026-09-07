"""Constraints shared by model-based controllers."""

import numpy as np
import pink
from pink.tasks import Task


class LockedVelocityConstraint(Task):
    """Hard equality constraint that locks selected tangent coordinates."""

    def __init__(self, model_nv: int, locked_indices: np.ndarray):
        super().__init__(cost=1.0, gain=1.0, lm_damping=0.0)
        self._jacobian = np.eye(model_nv)[locked_indices]

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        return np.zeros(self._jacobian.shape[0])

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        return self._jacobian

    def __repr__(self) -> str:
        return f"LockedVelocityConstraint(size={self._jacobian.shape[0]})"


def lock_uncontrolled_velocities(
    model_nv: int, controlled_indices: np.ndarray
) -> list[LockedVelocityConstraint]:
    """Create a hard constraint that freezes every uncontrolled coordinate."""
    controlled = np.zeros(model_nv, dtype=bool)
    controlled[controlled_indices] = True
    locked_indices = np.flatnonzero(~controlled)
    if locked_indices.size == 0:
        return []
    return [LockedVelocityConstraint(model_nv, locked_indices)]
