"""Arm operational-space controller using Pink for 6-DOF task-space control.

This controller computes joint commands to achieve target task space poses using
the Pink inverse kinematics library. Mobile-base, wheel, and gripper coordinates
are hard-locked and owned by separate controllers.
"""

import time

import numpy as np
import pink
import pinocchio as pin
from numpy.typing import NDArray
from pink.barriers import SelfCollisionBarrier
from pink.limits import ConfigurationLimit, Limit, VelocityLimit
from pink.tasks import (
    DampingTask,
    FrameTask,
    LowAccelerationTask,
    RelativeFrameTask,
)
from pink.utils import process_collision_pairs

from humanoid.controllers.base import Controller
from humanoid.controllers.constraints import (
    BrakingAccelerationLimit,
    SelectedVelocityLimit,
    lock_uncontrolled_velocities,
)
from humanoid.controllers.manipulability import ManipulabilityTask
from humanoid.logger import get_logger
from humanoid.robots.base import Robot
from humanoid.types.controllers import (
    ControlResult,
    IKDiagnostics,
    OperationalSpaceConfig,
    TaskName,
)
from humanoid.types.robot import CartesianVelocity, CartesianVelocityLimits

logger = get_logger(__name__)


def advance_cartesian_pose(
    pose: pin.SE3,
    velocity: CartesianVelocity,
    duration_s: float,
) -> pin.SE3:
    """Advance a pose using command-frame linear and angular velocity."""
    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("Cartesian feedforward duration must be finite and non-negative.")
    return pin.SE3(
        pin.exp3(velocity.angular * duration_s) @ pose.rotation,
        pose.translation + velocity.linear * duration_s,
    )


def clamp_cartesian_velocity(
    velocity: CartesianVelocity,
    limits: CartesianVelocityLimits,
) -> CartesianVelocity:
    """Clamp linear and angular vector norms to configured tool limits."""

    def clamp_norm(vector: NDArray[np.float64], limit: float) -> NDArray[np.float64]:
        norm = float(np.linalg.norm(vector))
        return vector if norm <= limit else vector * (limit / norm)

    return CartesianVelocity(
        linear=clamp_norm(velocity.linear, limits.linear),
        angular=clamp_norm(velocity.angular, limits.angular),
    )


class OperationalSpaceController(Controller[pin.SE3]):
    """Control only the arm joints for a 6-DOF tool-space target."""

    def __init__(
        self,
        robot: Robot,
        config: OperationalSpaceConfig | None = None,
    ):
        """Initialize the operational space controller.

        Args:
            robot: Robot instance containing the model and data
            config: Controller configuration parameters
        """
        self.config = config or OperationalSpaceConfig()
        self.robot = robot
        self._data = robot.model.createData()
        self._previous_velocity = np.zeros(robot.model.nv)

        # Defer configuration initialization until first state update
        self.configuration: pink.Configuration | None = None

        # Create tasks dictionary
        self.tasks = {}

        # Track the tool in world for fixed robots and relative to the physical base
        # for mobile robots. Relative tracking removes the planar root coordinates
        # from the task while preserving the same tool-command convention.
        robot.assert_frame_exists(robot.config.tool.frame)
        base_config = robot.config.base
        if base_config is None:
            self.tasks[TaskName.TOOL] = FrameTask(
                robot.config.tool.frame,
                position_cost=self.config.tool_position_cost,
                orientation_cost=self.config.tool_orientation_cost,
            )
        else:
            robot.assert_frame_exists(base_config.frame)
            self.tasks[TaskName.TOOL] = RelativeFrameTask(
                robot.config.tool.frame,
                base_config.frame,
                position_cost=self.config.tool_position_cost,
                orientation_cost=self.config.tool_orientation_cost,
            )

        self.controlled_joint_indices = robot.get_arm_joint_indices()
        self.controlled_q_indices = np.asarray(
            robot.get_joint_position_indices(self.controlled_joint_indices), dtype=int
        )
        self.controlled_v_indices = np.asarray(
            robot.get_joint_velocity_indices(self.controlled_joint_indices), dtype=int
        )
        self._constraints = lock_uncontrolled_velocities(robot.model.nv, self.controlled_v_indices)

        # Pink damping and low-acceleration tasks omit floating-root coordinates but include
        # every other model joint. Mask wheel and gripper joints automatically so
        # operational-space configuration only tunes arm behavior.
        root_v_slice = robot.get_root_v_slice()
        root_nv = 0 if root_v_slice is None else root_v_slice.stop - root_v_slice.start
        self._root_nv = root_nv
        self._arm_mask = np.zeros(robot.model.nv - root_nv)
        self._arm_task_indices = self.controlled_v_indices - root_nv
        self._arm_mask[self._arm_task_indices] = 1.0
        self._configure_regularization_tasks()
        self._configure_motion_limits()

        # Initialize barriers
        self.barriers = []

        if self.config.avoid_collisions:
            # NOTE: must update robot with collision data returned by process_collision_pairs
            self.robot.collision_data = process_collision_pairs(
                self.robot.model, self.robot.collision_model, str(self.robot.srdf_path)
            )

            # Create self-collision barrier
            collision_barrier = SelfCollisionBarrier(
                n_collision_pairs=len(self.robot.collision_model.collisionPairs),
                d_min=self.config.min_collision_distance,
                safe_displacement_gain=self.config.collision_safe_displacement_gain,
            )
            self.barriers.append(collision_barrier)
            logger.info(
                "Collision avoidance enabled "
                f"with {len(self.robot.collision_model.collisionPairs)} collision pairs"
            )
        else:
            logger.info("Collision avoidance disabled")

    def _arm_cost_mask(self, mask: np.ndarray | float, label: str) -> np.ndarray:
        values = np.asarray(mask, dtype=float)
        if values.ndim == 0:
            return self._arm_mask * values
        if values.shape == self.controlled_v_indices.shape:
            result = np.zeros_like(self._arm_mask)
            result[self._arm_task_indices] = values
            return result
        if values.shape == self._arm_mask.shape:
            return self._arm_mask * values
        raise ValueError(
            f"{label} must be a scalar, one value per arm velocity "
            f"({len(self.controlled_v_indices)}), or one value per non-root "
            f"velocity ({len(self._arm_mask)}); received {values.shape}."
        )

    def _arm_limit_values(self, configured: np.ndarray | float, label: str) -> np.ndarray:
        values = np.asarray(configured, dtype=float)
        if values.ndim == 0:
            return np.full(len(self.controlled_v_indices), float(values))
        if values.shape == self.controlled_v_indices.shape:
            return values
        raise ValueError(
            f"{label} must be a scalar or one value per arm velocity "
            f"({len(self.controlled_v_indices)}); received {values.shape}."
        )

    def _configure_regularization_tasks(self) -> None:
        if self.config.manipulability_cost > 0.0:
            self.tasks[TaskName.MANIPULABILITY] = ManipulabilityTask(
                model=self.robot.model,
                frame=self.robot.config.tool.frame,
                controlled_v_indices=self.controlled_v_indices,
                cost=self.config.manipulability_cost,
                regularization=self.config.manipulability_regularization,
            )
        self.tasks[TaskName.DAMPING] = DampingTask(
            cost=(
                self.config.damping_cost
                * self._arm_cost_mask(self.config.damping_mask, "damping_mask")
            )  # ty:ignore[invalid-argument-type]
        )

        self._low_acceleration_task: LowAccelerationTask | None = None
        if self.config.low_acceleration_cost <= 0.0:
            return
        self._low_acceleration_task = LowAccelerationTask(
            cost=(
                self.config.low_acceleration_cost
                * self._arm_cost_mask(
                    self.config.low_acceleration_mask,
                    "low_acceleration_mask",
                )
            )  # ty:ignore[invalid-argument-type]
        )
        # Pink's posture-style Jacobian excludes floating-root velocities,
        # so seed the task with a matching zero displacement before its first solve.
        self._low_acceleration_task.set_last_integration(
            np.zeros(self.robot.model.nv - self._root_nv),
            self.config.dt,
        )
        self.tasks[TaskName.LOW_ACCELERATION] = self._low_acceleration_task

    def _configure_motion_limits(self) -> None:
        self._acceleration_limit: BrakingAccelerationLimit | None = None
        additional_limits: list[Limit] = []
        if self.config.joint_velocity_limit is not None:
            additional_limits.append(
                SelectedVelocityLimit(
                    self.robot.model.nv,
                    self.controlled_v_indices,
                    self._arm_limit_values(
                        self.config.joint_velocity_limit,
                        "joint_velocity_limit",
                    ),
                )
            )
        if self.config.joint_acceleration_limit is not None:
            acceleration_limits = np.zeros(self.robot.model.nv)
            acceleration_limits[self.controlled_v_indices] = self._arm_limit_values(
                self.config.joint_acceleration_limit,
                "joint_acceleration_limit",
            )
            self._acceleration_limit = BrakingAccelerationLimit(
                self.robot.model,
                acceleration_limits,
                position_margin=self.config.joint_position_margin,
            )
            additional_limits.append(self._acceleration_limit)

        self._limits = None
        if additional_limits:
            self._limits = [
                # The braking envelope already steers away from position limits.
                # Pink's default 0.5 gain can demand a faster stop than the hard
                # acceleration bound allows. Keep only the full-step position bound.
                ConfigurationLimit(
                    self.robot.model,
                    config_limit_gain=1.0 if self._acceleration_limit is not None else 0.5,
                ),
                VelocityLimit(self.robot.model),
                *additional_limits,
            ]

    def update_state(self, q: np.ndarray) -> None:
        """Update the robot configuration state.

        Args:
            q: Joint configuration vector (nq,)
        """
        if self.configuration is None:
            collision_model = self.robot.collision_model if self.config.avoid_collisions else None
            collision_data = self.robot.collision_data if self.config.avoid_collisions else None

            # Initialize configuration on first state update
            self.configuration = pink.Configuration(
                self.robot.model,
                self._data,
                q,
                collision_model=collision_model,
                collision_data=collision_data,
            )
            logger.debug(f"Initialized controller configuration with state: q={q}")
        else:
            self.configuration.update(q)

    def compute_control(
        self,
        target: pin.SE3,
        dt: float | None = None,
        target_velocity: CartesianVelocity | None = None,
    ) -> ControlResult:
        """Compute arm motion to achieve a target tool pose.

        Uses Pink's differential inverse kinematics solver with:
        1. Track the target pose in task space (with optional masking via costs).
        2. Increase arm manipulability with a lower-weight soft objective.
        3. Minimize joint velocities and optionally changes in velocity.

        These objectives are weighted together rather than strictly prioritized.

        Args:
            target: Target 6-DOF pose (SE3) for the end-effector.
            dt: Controller integration period. Defaults to the configured period.
            target_velocity: Optional command-frame Cartesian reference velocity. The
                velocity is clamped to the tool limits and applied as a one-step pose
                preview, which supplies feedforward to Pink's pose task.

        Returns:
            Full-model result in which only arm coordinates can change.

        Raises:
            RuntimeError: If configuration has not been initialized via update_state()
        """
        if self.configuration is None:
            raise RuntimeError(
                "Controller configuration not initialized. "
                "Call update_state() with robot state first."
            )

        dt = self.config.dt if dt is None else dt
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Controller timestep must be positive and finite.")

        # Pink's pose task closes its target error over this integration step.
        # Previewing a moving reference by one step is therefore equivalent to
        # adding its Cartesian velocity as feedforward to the task objective.
        if target_velocity is not None:
            target_velocity = clamp_cartesian_velocity(
                target_velocity,
                self.robot.config.tool.velocity_limits,
            )
            target = advance_cartesian_pose(target, target_velocity, dt)

        # Set the target for the end-effector task
        self.tasks[TaskName.TOOL].set_target(target)

        # Solve inverse kinematics using Pink
        # Pink stores a previous displacement and interprets it using this solve's
        # dt. Re-express the last velocity over the current period so scheduling
        # jitter cannot change the implied velocity or defeat acceleration limits.
        self._set_previous_velocity(self._previous_velocity, dt)
        velocity = np.zeros(self.robot.model.nv)
        started_s = time.perf_counter()
        error = None
        try:
            solved_velocity = pink.solve_ik(
                self.configuration,
                self.tasks.values(),
                dt,
                solver=self.config.solver,
                limits=self._limits,
                barriers=self.barriers,
                constraints=self._constraints,
            )
            velocity[self.controlled_v_indices] = solved_velocity[self.controlled_v_indices]
            self.configuration.integrate_inplace(velocity, dt)
            self._set_previous_velocity(velocity, dt)
        except Exception as e:
            # TODO: try to get unstuck if we are at limits
            logger.error(f"Encountered exception: {e}")
            error = str(e)
            self._set_previous_velocity(velocity, dt)

        return ControlResult(
            q=self.configuration.q.copy(),
            v=velocity,
            diagnostics=IKDiagnostics(
                succeeded=error is None,
                duration_s=time.perf_counter() - started_s,
                error=error,
            ),
        )

    def reset_motion(self) -> None:
        """Reset stateful velocity smoothing when control is disengaged."""
        self._set_previous_velocity(np.zeros(self.robot.model.nv), self.config.dt)

    def _set_previous_velocity(self, velocity: np.ndarray, dt: float) -> None:
        self._previous_velocity = velocity.copy()
        if self._acceleration_limit is not None:
            self._acceleration_limit.set_last_integration(velocity, dt)
        if self._low_acceleration_task is not None:
            # Pink posture-style tasks omit the floating-root coordinates.
            self._low_acceleration_task.set_last_integration(velocity[self._root_nv :], dt)
