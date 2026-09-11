"""Constraints shared by model-based controllers."""

import numpy as np
import pink
from pink.limits import Limit
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


class SelectedVelocityLimit(Limit):
    """Bound the tangent velocity of a selected set of model coordinates."""

    def __init__(
        self,
        model_nv: int,
        indices: np.ndarray,
        velocity_limits: np.ndarray,
    ) -> None:
        self._projection = np.eye(model_nv)[indices]
        self._velocity_limits = velocity_limits.copy()

    def compute_qp_inequalities(
        self,
        configuration: pink.Configuration,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del configuration
        matrix = np.vstack((self._projection, -self._projection))
        displacement_limits = dt * self._velocity_limits
        vector = np.concatenate((displacement_limits, displacement_limits))
        return matrix, vector
