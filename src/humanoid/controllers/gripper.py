"""Direct joint-position controller for robot grippers."""

import numpy as np
from numpy.typing import NDArray

from humanoid.controllers.base import Controller
from humanoid.robots.base import Robot
from humanoid.types.controllers import ControlResult


class GripperController(Controller[NDArray[np.float64]]):
    """Apply gripper targets without coupling them to an arm controller."""

    def __init__(self, robot: Robot):
        controlled_joint_indices = robot.get_gripper_joint_indices()
        if not controlled_joint_indices:
            raise ValueError("Gripper controller requires configured gripper joints.")
        self.robot = robot
        self.controlled_joint_indices = controlled_joint_indices
        self.controlled_q_indices = np.asarray(
            robot.get_joint_position_indices(self.controlled_joint_indices), dtype=int
        )
        self.controlled_v_indices = np.asarray(
            robot.get_joint_velocity_indices(self.controlled_joint_indices), dtype=int
        )
        position_limits = np.asarray(robot.get_gripper_limits())
        self._lower_position_limits = position_limits[:, 0]
        self._upper_position_limits = position_limits[:, 1]
        self.configuration: NDArray[np.float64] | None = None

    def update_state(self, q: NDArray[np.float64]) -> None:
        """Replace the controller's full-model configuration state."""
        if q.shape != (self.robot.model.nq,):
            raise ValueError(
                f"Gripper state must have {self.robot.model.nq} values; received {q.shape}."
            )
        if not np.isfinite(q).all():
            raise ValueError("Gripper state values must all be finite.")
        self.configuration = q.copy()

    def compute_control(
        self,
        target: NDArray[np.float64],
        dt: float | None = None,
    ) -> ControlResult:
        """Set one target position for each configured gripper joint."""
        if self.configuration is None:
            raise RuntimeError(
                "Controller configuration not initialized. "
                "Call update_state() with robot state first."
            )
        expected_shape = (len(self.controlled_joint_indices),)
        if target.shape != expected_shape:
            raise ValueError(
                f"Gripper target must have shape {expected_shape}; received {target.shape}."
            )
        if not np.isfinite(target).all():
            raise ValueError("Gripper target values must all be finite.")

        q = self.configuration.copy()
        bounded_target = np.clip(
            target,
            self._lower_position_limits,
            self._upper_position_limits,
        )
        velocity = np.zeros(self.robot.model.nv)
        if dt is not None:
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError("Controller timestep must be positive and finite.")
            position_delta = bounded_target - q[self.controlled_q_indices]
            velocity[self.controlled_v_indices] = np.clip(
                position_delta / dt,
                -self.robot.model.velocityLimit[self.controlled_v_indices],
                self.robot.model.velocityLimit[self.controlled_v_indices],
            )
        self.robot.set_gripper_positions(q, bounded_target)
        self.configuration = q
        return ControlResult(q=q.copy(), v=velocity)
