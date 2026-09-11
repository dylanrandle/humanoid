"""Analytic pose controller for an omniwheel mobile base."""

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.controllers.base import Controller
from humanoid.robots.base import Robot
from humanoid.robots.wheels import WheelKinematics
from humanoid.types.controllers import ControlResult, OmniwheelBaseConfig
from humanoid.types.robot import WheelType


class OmniwheelBaseController(Controller[pin.SE3]):
    """Drive a planar base with bounded proportional pose feedback."""

    def __init__(
        self,
        robot: Robot,
        config: OmniwheelBaseConfig | None = None,
    ):
        base_config = robot.config.base
        root_q_slice = robot.get_root_q_slice()
        root_v_slice = robot.get_root_v_slice()
        if base_config is None or root_q_slice is None or root_v_slice is None:
            raise ValueError("Omniwheel base controller requires a configured planar base.")
        if not robot.config.wheels:
            raise ValueError("Omniwheel base controller requires configured wheels.")
        if any(wheel.type is not WheelType.OMNI for wheel in robot.config.wheels):
            raise ValueError("Omniwheel base controller only supports omniwheels.")

        self.config = config or OmniwheelBaseConfig()
        self.robot = robot
        self.configuration: NDArray[np.float64] | None = None
        self.target_pose: pin.SE3 | None = None
        self._root_q_slice = root_q_slice
        self._root_v_slice = root_v_slice
        self._wheel_kinematics = WheelKinematics(robot)

        root_joint_indices = [0]
        wheel_joint_indices = robot.get_wheel_joint_indices()
        self.controlled_joint_indices = [
            *root_joint_indices,
            *wheel_joint_indices,
        ]
        self.controlled_q_indices = np.asarray(
            robot.get_joint_position_indices(self.controlled_joint_indices), dtype=int
        )
        self.controlled_v_indices = np.asarray(
            robot.get_joint_velocity_indices(self.controlled_joint_indices), dtype=int
        )
        self._wheel_v_indices = np.asarray(
            robot.get_joint_velocity_indices(wheel_joint_indices),
            dtype=int,
        )
        self._wheel_velocity_limits = robot.model.velocityLimit[self._wheel_v_indices]
        self._linear_velocity_limit = base_config.velocity_limits.linear
        self._angular_velocity_limit = base_config.velocity_limits.angular

    def update_state(self, q: NDArray[np.float64]) -> None:
        """Update the full robot state used to compute base motion."""
        if q.shape != (self.robot.model.nq,):
            raise ValueError(
                f"Configuration must have {self.robot.model.nq} values; received {q.shape}."
            )
        if not np.isfinite(q).all():
            raise ValueError("Configuration values must all be finite.")
        self.configuration = q.copy()

    def compute_control(self, target: pin.SE3, dt: float | None = None) -> ControlResult:
        """Compute bounded base and wheel motion toward ``target``."""
        if self.configuration is None:
            raise RuntimeError(
                "Controller configuration not initialized. "
                "Call update_state() with robot state first."
            )

        dt = self.config.dt if dt is None else dt
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Controller timestep must be positive and finite.")

        # Frame poses returned by Robot may alias its reusable Pinocchio data.
        # Copy the command before querying the current pose into that same data.
        target = target.copy()
        current_pose = self.robot.get_base_pose(self.configuration)
        if current_pose is None:  # Guard against an inconsistent robot configuration.
            raise RuntimeError("Controller robot no longer has a configured planar base.")

        target_in_base = current_pose.actInv(target)
        yaw_error = np.arctan2(
            target_in_base.rotation[1, 0],
            target_in_base.rotation[0, 0],
        )
        root_velocity = np.array(
            [
                self.config.position_gain * target_in_base.translation[0] / dt,
                self.config.position_gain * target_in_base.translation[1] / dt,
                self.config.orientation_gain * yaw_error / dt,
            ]
        )
        root_velocity[:2] = np.clip(
            root_velocity[:2],
            -self._linear_velocity_limit,
            self._linear_velocity_limit,
        )
        root_velocity[2] = np.clip(
            root_velocity[2],
            -self._angular_velocity_limit,
            self._angular_velocity_limit,
        )

        wheel_velocity = self._wheel_kinematics.compute_wheel_velocities(
            self.configuration,
            root_velocity,
        )
        wheel_scale = np.min(
            np.divide(
                self._wheel_velocity_limits,
                np.abs(wheel_velocity),
                out=np.full_like(wheel_velocity, np.inf),
                where=np.abs(wheel_velocity) > 0.0,
            )
        )
        if wheel_scale < 1.0:
            root_velocity *= wheel_scale
            wheel_velocity *= wheel_scale

        velocity = np.zeros(self.robot.model.nv)
        velocity[self._root_v_slice] = root_velocity
        velocity[self._wheel_v_indices] = wheel_velocity
        self.configuration = pin.integrate(
            self.robot.model,
            self.configuration,
            velocity * dt,
        )
        self.target_pose = target

        return ControlResult(q=self.configuration.copy(), v=velocity)
