"""Wheel geometry and model-based kinematics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pinocchio as pin

from humanoid.types.robot import WheelType

if TYPE_CHECKING:
    from humanoid.robots.base import Robot


class WheelKinematics:
    """Map between wheel rates and instantaneous body-frame root velocity."""

    def __init__(self, robot: Robot):
        root_v_slice = robot.get_root_v_slice()
        if root_v_slice is None:
            raise ValueError("Wheel kinematics requires a planar root joint.")
        if not robot.config.wheels:
            raise ValueError("Wheel kinematics requires configured wheels.")

        self._robot = robot
        self._root_v_slice = root_v_slice
        self._data = robot.model.createData()
        self._wheel_v_indices = np.asarray(
            robot.get_joint_velocity_indices(robot.get_wheel_joint_indices()),
            dtype=int,
        )
        if self._wheel_v_indices.size != len(robot.config.wheels):
            raise ValueError("Wheel kinematics requires one velocity coordinate per wheel.")

        self._wheels = []
        for wheel in robot.config.wheels:
            robot.assert_frame_exists(wheel.frame)
            robot.assert_frame_exists(wheel.floor_frame)
            self._wheels.append(
                (
                    wheel.type,
                    robot.get_frame_id(wheel.frame),
                    robot.get_frame_id(wheel.floor_frame),
                    wheel.radius,
                )
            )

    def _constraint_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Build the no-slip wheel constraint Jacobian from the current model pose."""
        if q.shape != (self._robot.model.nq,):
            raise ValueError(
                f"Measured configuration must have {self._robot.model.nq} values; "
                f"received {q.shape}."
            )
        if not np.isfinite(q).all():
            raise ValueError("Measured wheel state values must all be finite.")

        model = self._robot.model
        pin.computeJointJacobians(model, self._data, q)
        pin.updateFramePlacements(model, self._data)

        constraint_rows = []
        for wheel_type, wheel_frame_id, floor_frame_id, wheel_radius in self._wheels:
            wheel_in_world = self._data.oMf[wheel_frame_id]
            floor_in_world = self._data.oMf[floor_frame_id]
            wheel_in_floor = floor_in_world.actInv(wheel_in_world)

            rim_position_in_floor = wheel_in_floor.translation + np.array([0.0, 0.0, -wheel_radius])
            rim_in_floor = pin.SE3(np.eye(3), rim_position_in_floor)
            wheel_in_rim = rim_in_floor.actInv(wheel_in_floor)

            wheel_jacobian = pin.getFrameJacobian(
                model,
                self._data,
                wheel_frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            rim_jacobian = wheel_in_rim.action @ wheel_jacobian
            horizontal_row_count = 1 if wheel_type is WheelType.OMNI else 2
            constraint_rows.append(rim_jacobian[:horizontal_row_count])

        return np.vstack(constraint_rows)

    def estimate_root_velocity(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Estimate the root twist from measured configuration and joint rates."""
        if v.shape != (self._robot.model.nv,):
            raise ValueError(
                f"Measured velocity must have {self._robot.model.nv} values; received {v.shape}."
            )
        if not np.isfinite(v).all():
            raise ValueError("Measured wheel state values must all be finite.")

        constraint_jacobian = self._constraint_jacobian(q)
        root_jacobian = constraint_jacobian[:, self._root_v_slice]
        if np.linalg.matrix_rank(root_jacobian) < root_jacobian.shape[1]:
            raise RuntimeError("Wheel geometry cannot determine planar root velocity.")

        wheel_jacobian = constraint_jacobian[:, self._wheel_v_indices]
        constraint_velocity = wheel_jacobian @ v[self._wheel_v_indices]
        root_velocity, *_ = np.linalg.lstsq(
            root_jacobian,
            -constraint_velocity,
            rcond=None,
        )
        return root_velocity

    def compute_wheel_velocities(
        self,
        q: np.ndarray,
        root_velocity: np.ndarray,
    ) -> np.ndarray:
        """Compute wheel rates that realize a requested body-frame root twist."""
        expected_shape = (self._root_v_slice.stop - self._root_v_slice.start,)
        if root_velocity.shape != expected_shape:
            raise ValueError(
                f"Root velocity must have {expected_shape[0]} values; "
                f"received {root_velocity.shape}."
            )
        if not np.isfinite(root_velocity).all():
            raise ValueError("Requested root velocity values must all be finite.")

        constraint_jacobian = self._constraint_jacobian(q)
        root_jacobian = constraint_jacobian[:, self._root_v_slice]
        wheel_jacobian = constraint_jacobian[:, self._wheel_v_indices]
        if np.linalg.matrix_rank(root_jacobian) < root_jacobian.shape[1]:
            raise RuntimeError("Wheel geometry cannot control planar root velocity.")
        if np.linalg.matrix_rank(wheel_jacobian) < wheel_jacobian.shape[1]:
            raise RuntimeError("Wheel geometry cannot realize planar root velocity.")

        wheel_velocity, *_ = np.linalg.lstsq(
            wheel_jacobian,
            -(root_jacobian @ root_velocity),
            rcond=None,
        )
        return wheel_velocity
