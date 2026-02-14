"""Operational space controller using Pinocchio for 6-DOF task space control.

This controller computes joint commands to achieve target task space poses.
"""

from dataclasses import dataclass

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.logger import get_logger
from humanoid.robots.base import Robot

logger = get_logger(__name__)


@dataclass
class OperationalSpaceConfig:
    """Configuration parameters for the operational space controller."""

    # Task space gains
    kp_position: float = 100.0  # Position error gain
    kp_orientation: float = 50.0  # Orientation error gain
    kd_position: float = 20.0  # Position damping
    kd_orientation: float = 10.0  # Orientation damping

    # Control loop
    dt: float = 0.01  # Control timestep (seconds)

    # Velocity and acceleration limits (per axis)
    max_linear_velocity: float = 1.0  # Maximum linear velocity per axis (m/s)
    max_angular_velocity: float = 2.0  # Maximum angular velocity per axis (rad/s)
    max_linear_acceleration: float = 10.0  # Maximum linear acceleration per axis (m/s^2)
    max_angular_acceleration: float = 20.0  # Maximum angular acceleration per axis (rad/s^2)


class OperationalSpaceController:
    """Operational space controller for 6-DOF task space control."""

    def __init__(
        self,
        robot: Robot,
        end_effector_frame: str,
        config: OperationalSpaceConfig | None = None,
    ):
        """Initialize the operational space controller.

        Args:
            robot: Robot instance containing the model and data
            end_effector_frame: Name of the end-effector frame in the URDF
            config: Controller configuration parameters
        """
        self.config = config or OperationalSpaceConfig()
        self.robot = robot

        # Use robot's model and data
        self.model = robot.model
        self.data = robot.data

        # Get end-effector frame ID
        if not self.model.existFrame(end_effector_frame):
            raise ValueError(f"Frame '{end_effector_frame}' not found in URDF")
        self.ee_frame_id = self.model.getFrameId(end_effector_frame)

        # Store dimensions
        self.nq = self.model.nq  # Number of joint positions
        self.nv = self.model.nv  # Number of joint velocities

        # State tracking
        self.q_current: NDArray[np.float64] | None = None
        self.v_current: NDArray[np.float64] | None = None
        self.prev_task_velocity: NDArray[np.float64] | None = None

    def update_state(self, q: NDArray[np.float64], v: NDArray[np.float64]) -> None:
        """Update the current robot state.

        Args:
            q: Current joint positions (nq,)
            v: Current joint velocities (nv,)
        """
        self.q_current = q.copy()
        self.v_current = v.copy()

        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)

    def compute_task_error(
        self, target_pose: pin.SE3
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute task space error between current and target pose.

        Args:
            target_pose: Target 6-DOF pose (SE3)

        Returns:
            position_error: 3D position error (3,)
            orientation_error: 3D orientation error in log coordinates (3,)
        """
        if self.q_current is None:
            raise RuntimeError("State not initialized. Call update_state first.")

        # Get current end-effector pose
        current_pose = self.data.oMf[self.ee_frame_id]

        # Position error
        position_error = target_pose.translation - current_pose.translation

        # Orientation error (log map of rotation difference)
        rotation_error = current_pose.rotation.T @ target_pose.rotation
        orientation_error = pin.log3(rotation_error)

        return position_error, orientation_error

    def compute_jacobian(self) -> NDArray[np.float64]:
        """Compute the task space Jacobian for the end-effector.

        Returns:
            J: Task space Jacobian (6, nv)
        """
        if self.q_current is None:
            raise RuntimeError("State not initialized. Call update_state first.")

        # Compute frame Jacobian in local frame
        return pin.computeFrameJacobian(
            self.model,
            self.data,
            self.q_current,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

    def _clip_task_velocity(self, task_velocity: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip task space velocity to configured limits (per axis).

        Args:
            task_velocity: 6D task space velocity [linear, angular]

        Returns:
            Clipped task space velocity
        """
        linear_vel = np.clip(
            task_velocity[:3], -self.config.max_linear_velocity, self.config.max_linear_velocity
        )
        angular_vel = np.clip(
            task_velocity[3:], -self.config.max_angular_velocity, self.config.max_angular_velocity
        )
        return np.concatenate([linear_vel, angular_vel])

    def _clip_task_acceleration(
        self, task_acceleration: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Clip task space acceleration to configured limits (per axis).

        Args:
            task_acceleration: 6D task space acceleration [linear, angular]

        Returns:
            Clipped task space acceleration
        """
        linear_accel = np.clip(
            task_acceleration[:3],
            -self.config.max_linear_acceleration,
            self.config.max_linear_acceleration,
        )
        angular_accel = np.clip(
            task_acceleration[3:],
            -self.config.max_angular_acceleration,
            self.config.max_angular_acceleration,
        )
        return np.concatenate([linear_accel, angular_accel])

    def compute_control(
        self, target_pose: pin.SE3, target_velocity: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Compute joint velocity commands to achieve target task space pose.

        Args:
            target_pose: Target 6-DOF pose (SE3)
            target_velocity: Optional target task space velocity (6,)

        Returns:
            dq: Joint velocity commands (nv,)
        """
        if self.q_current is None or self.v_current is None:
            raise RuntimeError("State not initialized. Call update_state first.")

        if target_velocity is None:
            target_velocity = np.zeros(6)

        # Compute task space error
        pos_error, ori_error = self.compute_task_error(target_pose)
        task_error = np.concatenate([pos_error, ori_error])

        # Compute desired task space velocity (PD control)
        Kp = np.diag([self.config.kp_position] * 3 + [self.config.kp_orientation] * 3)
        Kd = np.diag([self.config.kd_position] * 3 + [self.config.kd_orientation] * 3)

        # Get current task velocity
        J = self.compute_jacobian()
        current_task_velocity = J @ self.v_current

        # Desired task velocity with PD control
        desired_task_velocity = target_velocity + Kp @ task_error - Kd @ current_task_velocity

        # Apply velocity limits
        desired_task_velocity = self._clip_task_velocity(desired_task_velocity)

        # Apply acceleration limits
        if self.prev_task_velocity is not None:
            task_acceleration = (desired_task_velocity - self.prev_task_velocity) / self.config.dt
            task_acceleration = self._clip_task_acceleration(task_acceleration)
            desired_task_velocity = self.prev_task_velocity + task_acceleration * self.config.dt

        # Store for next iteration
        self.prev_task_velocity = desired_task_velocity.copy()

        # Compute joint velocities using pseudo-inverse
        J_pinv = np.linalg.pinv(J)
        return J_pinv @ desired_task_velocity

    def compute_joint_commands(
        self,
        q_current: NDArray[np.float64],
        v_current: NDArray[np.float64],
        target_pose: pin.SE3,
        target_velocity: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """High-level interface to compute joint position commands.

        Args:
            q_current: Current joint positions (nq,)
            v_current: Current joint velocities (nv,)
            target_pose: Target 6-DOF pose (SE3)
            target_velocity: Optional target task space velocity (6,)

        Returns:
            q_cmd: Joint position commands (nq,)
        """
        # Update state
        self.update_state(q_current, v_current)

        # Compute joint velocity commands
        dq = self.compute_control(target_pose, target_velocity)

        # Integrate to get position commands
        q_cmd = q_current + dq * self.config.dt

        # Clamp to joint limits
        return np.clip(q_cmd, self.model.lowerPositionLimit, self.model.upperPositionLimit)
