"""Constraints shared by model-based controllers."""

import numpy as np
import pink
import pinocchio as pin
from pink.limits import AccelerationLimit, Limit
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


class BrakingAccelerationLimit(AccelerationLimit):
    """Leave room for this integration step and subsequent maximum braking.

    For distance d to a joint bound, require v*dt + v**2/(2*a) <= d.
    Pink's continuous stopping bound omits the first displacement, which can
    leave the next QP unable to satisfy both position and acceleration limits.
    The nonnegative root below preserves a feasible braking step even when dt
    changes. A margin keeps position targets away from physical joint stops.
    """

    def __init__(
        self, model: pin.Model, acceleration_limit: np.ndarray, position_margin: float = 0.0
    ) -> None:
        super().__init__(model, acceleration_limit)
        self.position_margin = position_margin

    def compute_qp_inequalities(
        self, configuration: pink.Configuration, dt: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.projection_matrix is None:
            return None
        upper = pin.difference(self.model, configuration.q, self.model.upperPositionLimit)
        lower = pin.difference(self.model, self.model.lowerPositionLimit, configuration.q)

        def speed_bound(distance: np.ndarray) -> np.ndarray:
            distance = np.maximum(distance[self.indices] - self.position_margin, 0.0)
            a_dt = self.a_max * dt
            return np.sqrt(a_dt**2 + 2 * self.a_max * distance) - a_dt

        acceleration_step = self.a_max * dt**2
        matrix = np.vstack((self.projection_matrix, -self.projection_matrix))
        vector = np.concatenate(
            (
                np.minimum(acceleration_step + self.Delta_q_prev, dt * speed_bound(upper)),
                np.minimum(acceleration_step - self.Delta_q_prev, dt * speed_bound(lower)),
            )
        )
        return matrix, vector
