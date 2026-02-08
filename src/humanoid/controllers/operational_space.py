"""Operational space controller using Pinocchio for 6-DOF task space control.

This controller computes joint commands to achieve target task space poses while:
1. Attempting to achieve the target 6-DOF pose
2. Avoiding self-collisions
3. Penalizing proximity to joint limits
4. Limiting maximum task space velocity/acceleration
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray
from qpsolvers import solve_qp

from humanoid.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OperationalSpaceConfig:
    """Configuration parameters for the operational space controller."""

    # Task space gains
    kp_position: float = 100.0  # Position error gain
    kp_orientation: float = 50.0  # Orientation error gain
    kd_position: float = 20.0  # Position damping
    kd_orientation: float = 10.0  # Orientation damping

    # Linear velocity and acceleration limits
    max_linear_velocity: float = 1.0  # m/s
    max_linear_acceleration: float = 2.0  # m/s^2

    # Angular velocity and acceleration limits
    max_angular_velocity: float = np.pi  # rad/s
    max_angular_acceleration: float = 2.0 * np.pi  # rad/s^2

    # Joint limit avoidance
    joint_limit_margin: float = 0.1  # Margin from joint limits (radians)
    joint_limit_gain: float = 10.0  # Gain for joint limit repulsion

    # Self-collision avoidance
    collision_margin: float = 0.05  # Minimum distance to maintain (meters)
    collision_gain: float = 50.0  # Gain for collision avoidance
    collision_eps: float = 1e-6  # Tolerance for collision avoidance

    # QP solver parameters
    qp_weight_task: float = 1.0  # Weight for task tracking
    qp_weight_regularization: float = 1e-6  # Regularization weight
    qp_weight_joint_limits: float = 0.5  # Weight for joint limit avoidance
    qp_weight_collision: float = 1.0  # Weight for collision avoidance

    # Control loop
    dt: float = 0.01  # Control timestep (seconds)


class OperationalSpaceController:
    """Operational space controller for 6-DOF task space control."""

    def __init__(
        self,
        urdf_path: Path,
        end_effector_frame: str,
        config: OperationalSpaceConfig | None = None,
        package_dirs: list[Path] | None = None,
    ):
        """Initialize the operational space controller.

        Args:
            urdf_path: Path to the robot URDF file
            end_effector_frame: Name of the end-effector frame in the URDF
            config: Controller configuration parameters
        """
        self.config = config or OperationalSpaceConfig()
        self.urdf_path = urdf_path
        self.package_dirs = package_dirs

        # Load robot model
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        # Setup collision model for self-collision detection
        self.collision_model = pin.buildGeomFromUrdf(
            self.model,
            self.urdf_path,
            pin.GeometryType.COLLISION,
            package_dirs=self.package_dirs,
        )
        self.collision_data = self.collision_model.createData()

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
        self.task_velocity_prev = np.zeros(6)
        self.task_acceleration_prev = np.zeros(6)

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

    def compute_joint_limit_gradient(self) -> NDArray[np.float64]:
        """Compute gradient for joint limit avoidance using potential field.

        Returns:
            grad: Gradient of joint limit potential (nq,)
        """
        if self.q_current is None:
            raise RuntimeError("State not initialized. Call update_state first.")

        grad = np.zeros(self.nq)
        margin = self.config.joint_limit_margin

        for i in range(self.nq):
            q_lower = self.model.lowerPositionLimit[i]
            q_upper = self.model.upperPositionLimit[i]
            q = self.q_current[i]

            # Distance to lower limit
            dist_lower = q - q_lower
            if dist_lower < margin:
                # Repulsive potential: 0.5 * k * (1/d - 1/d0)^2
                grad[i] -= self.config.joint_limit_gain * (1.0 / dist_lower - 1.0 / margin)

            # Distance to upper limit
            dist_upper = q_upper - q
            if dist_upper < margin:
                grad[i] += self.config.joint_limit_gain * (1.0 / dist_upper - 1.0 / margin)

        return grad

    def compute_collision_avoidance_gradient(self) -> NDArray[np.float64]:
        """Compute gradient for self-collision avoidance.

        Returns:
            grad: Gradient for collision avoidance in joint space (nv,)
        """
        if self.q_current is None:
            raise RuntimeError("State not initialized. Call update_state first.")

        # Update collision geometry
        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.collision_data, self.q_current
        )

        # Compute all collision distances
        pin.computeDistances(
            self.model, self.data, self.collision_model, self.collision_data, self.q_current
        )

        grad = np.zeros(self.nv)

        # Iterate through collision pairs
        for k in range(len(self.collision_model.collisionPairs)):
            dist_result = self.collision_data.distanceResults[k]
            distance = dist_result.min_distance

            # Only consider pairs within collision margin
            if distance < self.config.collision_margin:
                pair = self.collision_model.collisionPairs[k]
                geom1_id = pair.first
                geom2_id = pair.second

                # Get parent joint IDs
                joint1_id = self.collision_model.geometryObjects[geom1_id].parentJoint
                joint2_id = self.collision_model.geometryObjects[geom2_id].parentJoint

                # Compute Jacobian of distance
                # Simplified: use witness points to compute gradient
                witness1 = dist_result.getNearestPoint1()
                witness2 = dist_result.getNearestPoint2()

                # Direction from geom2 to geom1
                direction = witness1 - witness2
                if np.linalg.norm(direction) > self.config.collision_eps:
                    direction = direction / np.linalg.norm(direction)

                    # Compute Jacobians at witness points
                    J1 = pin.computeFrameJacobian(
                        self.model,
                        self.data,
                        self.q_current,
                        joint1_id,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )[:3, :]  # Position part only

                    J2 = pin.computeFrameJacobian(
                        self.model,
                        self.data,
                        self.q_current,
                        joint2_id,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )[:3, :]  # Position part only

                    # Gradient: repulsive force proportional to penetration
                    penetration = self.config.collision_margin - distance
                    force_magnitude = self.config.collision_gain * penetration

                    # Add contribution to gradient
                    grad += force_magnitude * direction @ (J1 - J2)

        return grad

    def limit_task_velocity_acceleration(
        self, desired_task_velocity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Limit task space velocity and acceleration.

        Args:
            desired_task_velocity: Desired task space velocity (6,)

        Returns:
            limited_velocity: Velocity limited by max velocity and acceleration (6,)
        """
        # Separate position and orientation components
        v_pos = desired_task_velocity[:3]
        v_ori = desired_task_velocity[3:]

        # Limit linear velocity
        v_pos_norm = np.linalg.norm(v_pos)
        if v_pos_norm > self.config.max_linear_velocity:
            v_pos = v_pos * (self.config.max_linear_velocity / v_pos_norm)

        # Limit angular velocity
        v_ori_norm = np.linalg.norm(v_ori)
        if v_ori_norm > self.config.max_angular_velocity:
            v_ori = v_ori * (self.config.max_angular_velocity / v_ori_norm)

        limited_velocity = np.concatenate([v_pos, v_ori])

        # Limit acceleration
        task_acceleration = (limited_velocity - self.task_velocity_prev) / self.config.dt

        # Limit linear acceleration
        a_pos = task_acceleration[:3]
        a_pos_norm = np.linalg.norm(a_pos)
        if a_pos_norm > self.config.max_linear_acceleration:
            a_pos = a_pos * (self.config.max_linear_acceleration / a_pos_norm)
            limited_velocity[:3] = self.task_velocity_prev[:3] + a_pos * self.config.dt

        # Limit angular acceleration
        a_ori = task_acceleration[3:]
        a_ori_norm = np.linalg.norm(a_ori)
        if a_ori_norm > self.config.max_angular_acceleration:
            a_ori = a_ori * (self.config.max_angular_acceleration / a_ori_norm)
            limited_velocity[3:] = self.task_velocity_prev[3:] + a_ori * self.config.dt

        # Update previous values
        self.task_velocity_prev = limited_velocity.copy()
        self.task_acceleration_prev = (limited_velocity - self.task_velocity_prev) / self.config.dt

        return limited_velocity

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

        # Limit velocity and acceleration
        limited_task_velocity = self.limit_task_velocity_acceleration(desired_task_velocity)

        # Compute gradients for constraints
        joint_limit_grad = self.compute_joint_limit_gradient()
        collision_grad = self.compute_collision_avoidance_gradient()

        # Formulate QP problem: minimize ||J*dq - v_task||^2 + regularization + constraints
        # Variables: dq (joint velocities)
        # min: 0.5 * dq^T * H * dq + f^T * dq
        # s.t.: G * dq <= h (inequality constraints)

        # Cost function
        H = self.config.qp_weight_task * (J.T @ J) + self.config.qp_weight_regularization * np.eye(
            self.nv
        )
        f = -self.config.qp_weight_task * (J.T @ limited_task_velocity)

        # Add joint limit avoidance to cost (as soft constraint)
        f += self.config.qp_weight_joint_limits * joint_limit_grad

        # Add collision avoidance to cost (as soft constraint)
        f += self.config.qp_weight_collision * collision_grad

        # Inequality constraints: joint velocity limits
        # -v_max <= dq <= v_max
        v_max = self.model.velocityLimit
        G = np.vstack([np.eye(self.nv), -np.eye(self.nv)])
        h = np.concatenate([v_max, v_max])

        # Solve QP
        try:
            dq = solve_qp(H, f, G, h, solver="quadprog")
            if dq is None:
                raise RuntimeError("No solution from QP solver")
        except Exception as e:
            logger.error(f"Caught exception: {e}")
            # Fallback: use pseudo-inverse w/o constraints
            J_pinv = np.linalg.pinv(J)
            dq = J_pinv @ limited_task_velocity

        return dq

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
