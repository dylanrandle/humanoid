"""Arm-only manipulability objective for Pink inverse kinematics."""

import numpy as np
import pink
import pinocchio as pin
from pink.tasks import Task


class ManipulabilityTask(Task):
    """Encourage an increase in the regularized log-volume of the tool Jacobian.

    The measure is ``0.5 * log(det(J_arm @ J_arm.T + regularization * I))``.
    Its gradient favors configurations with greater Cartesian dexterity without
    counting motion from locked base, wheel, or gripper coordinates. Logarithmic
    scaling avoids a very small objective on short arms, and regularization keeps
    the measure and gradient finite at singularities and on arms with fewer than
    six degrees of freedom.

    Like Pink's other soft tasks, the cost competes with tracking and damping;
    this is not a strict null-space projection.
    """

    def __init__(
        self,
        model: pin.Model,
        frame: str,
        controlled_v_indices: np.ndarray,
        cost: float,
        regularization: float,
    ) -> None:
        super().__init__(cost=cost)
        self.frame = frame
        self.regularization = regularization
        self._controlled_v_indices = controlled_v_indices.copy()
        self._frame_id = model.getFrameId(frame)
        self._velocity_basis = np.eye(model.nv)[controlled_v_indices]
        # Derivative evaluations must not overwrite the live configuration's
        # kinematics or collision data used by the other tasks and barriers.
        self._data = model.createData()

    def _arm_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        return configuration.get_frame_jacobian(self.frame)[:, self._controlled_v_indices]

    def _regularized_gram(self, jacobian: np.ndarray) -> np.ndarray:
        return jacobian @ jacobian.T + self.regularization * np.eye(jacobian.shape[0])

    def compute_log_manipulability(self, configuration: pink.Configuration) -> float:
        """Return the regularized log-volume using only the controlled arm."""
        gram = self._regularized_gram(self._arm_jacobian(configuration))
        return 0.5 * float(np.linalg.slogdet(gram)[1])

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        """Request an increase of one log-volume unit per IK integration step."""
        return np.array([-1.0])

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        """Differentiate the log-volume in the arm's tangent coordinates."""
        jacobian = self._arm_jacobian(configuration)
        weighted_jacobian = np.linalg.solve(self._regularized_gram(jacobian), jacobian)
        gradient = np.zeros((1, configuration.model.nv))
        for index, velocity in zip(self._controlled_v_indices, self._velocity_basis, strict=True):
            # A unit joint velocity gives the directional derivative dJ/dq_i,
            # including the motion of the LOCAL reference frame.
            pin.computeJointJacobiansTimeVariation(
                configuration.model, self._data, configuration.q, velocity
            )
            pin.updateFramePlacements(configuration.model, self._data)
            derivative = pin.getFrameJacobianTimeVariation(
                configuration.model, self._data, self._frame_id, pin.ReferenceFrame.LOCAL
            )[:, self._controlled_v_indices]
            # d(log(det(G)) / 2) = trace(G^-1 J dJ^T).
            gradient[0, index] = np.sum(weighted_jacobian * derivative)
        return gradient

    def __repr__(self) -> str:
        return (
            f"ManipulabilityTask(frame={self.frame!r}, cost={self.cost}, "
            f"regularization={self.regularization})"
        )
