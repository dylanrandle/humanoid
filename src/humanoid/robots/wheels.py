"""Wheel geometry and model-based kinematics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pink
from pink.tasks import OmniwheelTask, RollingTask

from humanoid.types.robot import WheelType

if TYPE_CHECKING:
    from humanoid.robots.base import Robot


class WheelKinematics:
    """Map measured wheel motion to an instantaneous body-frame root velocity."""

    def __init__(self, robot: Robot):
        root_v_slice = robot.get_root_v_slice()
        if root_v_slice is None:
            raise ValueError("Wheel kinematics requires a planar root joint.")
        if not robot.config.wheels:
            raise ValueError("Wheel kinematics requires configured wheels.")

        self._robot = robot
        self._root_v_slice = root_v_slice
        self._configuration: pink.Configuration | None = None
        self._tasks = [
            (
                wheel.type,
                (OmniwheelTask if wheel.type is WheelType.OMNI else RollingTask)(
                    wheel.frame,
                    floor_frame=wheel.floor_frame,
                    wheel_radius=wheel.radius,
                    cost=1.0,
                ),
            )
            for wheel in robot.config.wheels
        ]

    def estimate_root_velocity(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Estimate the root twist from measured configuration and joint rates."""
        if q.shape != (self._robot.model.nq,):
            raise ValueError(
                f"Measured configuration must have {self._robot.model.nq} values; "
                f"received {q.shape}."
            )
        if v.shape != (self._robot.model.nv,):
            raise ValueError(
                f"Measured velocity must have {self._robot.model.nv} values; received {v.shape}."
            )
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("Measured wheel state values must all be finite.")

        if self._configuration is None:
            self._configuration = pink.Configuration(
                self._robot.model,
                self._robot.data,
                q,
            )
        else:
            self._configuration.update(q)

        constraint_rows = []
        for wheel_type, task in self._tasks:
            jacobian = task.compute_jacobian(self._configuration)
            horizontal_row_count = 1 if wheel_type is WheelType.OMNI else 2
            constraint_rows.append(jacobian[:horizontal_row_count])
        constraint_jacobian = np.vstack(constraint_rows)
        root_jacobian = constraint_jacobian[:, self._root_v_slice]
        if np.linalg.matrix_rank(root_jacobian) < root_jacobian.shape[1]:
            raise RuntimeError("Wheel geometry cannot determine planar root velocity.")

        measured_joint_velocity = v.copy()
        measured_joint_velocity[self._root_v_slice] = 0.0
        constraint_velocity = constraint_jacobian @ measured_joint_velocity
        root_velocity, *_ = np.linalg.lstsq(
            root_jacobian,
            -constraint_velocity,
            rcond=None,
        )
        return root_velocity
