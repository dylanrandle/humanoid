"""Arm operational-space controller using Pink for 6-DOF task-space control.

This controller computes joint commands to achieve target task space poses using
the Pink inverse kinematics library. Mobile-base, wheel, and gripper coordinates
are hard-locked and owned by separate controllers.
"""

from enum import StrEnum

import numpy as np
import pink
import pinocchio as pin
from pink.barriers import SelfCollisionBarrier
from pink.tasks import (
    DampingTask,
    FrameTask,
    PostureTask,
    RelativeFrameTask,
)
from pink.utils import process_collision_pairs

from humanoid.controllers.base import Controller
from humanoid.controllers.constraints import lock_uncontrolled_velocities
from humanoid.logger import get_logger
from humanoid.robots.base import Robot
from humanoid.types.controllers import ControlResult, OperationalSpaceConfig
from humanoid.types.homing import HomingPreset

logger = get_logger(__name__)


class TaskName(StrEnum):
    """Enum for task names used in the operational space controller."""

    TOOL = "tool"
    JOINT_CENTERING = "joint_centering"
    DAMPING = "damping"


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

        # Pink posture and damping tasks omit floating-root coordinates but include
        # every other model joint. Mask wheel and gripper joints automatically so
        # operational-space configuration only tunes arm behavior.
        root_v_slice = robot.get_root_v_slice()
        root_nv = 0 if root_v_slice is None else root_v_slice.stop - root_v_slice.start
        arm_mask = np.zeros(robot.model.nv - root_nv)
        arm_task_indices = self.controlled_v_indices - root_nv
        arm_mask[arm_task_indices] = 1.0

        def arm_cost_mask(mask: np.ndarray | float, label: str) -> np.ndarray:
            values = np.asarray(mask, dtype=float)
            if values.ndim == 0:
                return arm_mask * values
            if values.shape == self.controlled_v_indices.shape:
                result = np.zeros_like(arm_mask)
                result[arm_task_indices] = values
                return result
            if values.shape == arm_mask.shape:
                return arm_mask * values
            raise ValueError(
                f"{label} must be a scalar, one value per arm velocity "
                f"({len(self.controlled_v_indices)}), or one value per non-root "
                f"velocity ({len(arm_mask)}); received {values.shape}."
            )

        # Create posture task for null space control (joint centering)
        self.tasks[TaskName.JOINT_CENTERING] = PostureTask(
            cost=(
                self.config.joint_centering_cost
                * arm_cost_mask(self.config.joint_centering_mask, "joint_centering_mask")
            )  # ty:ignore[invalid-argument-type]
        )
        self.tasks[TaskName.JOINT_CENTERING].set_target(
            robot.config.homing_presets[HomingPreset.HOME]
        )

        # Create damping task for velocity minimization
        self.tasks[TaskName.DAMPING] = DampingTask(
            cost=(
                self.config.damping_cost * arm_cost_mask(self.config.damping_mask, "damping_mask")
            )  # ty:ignore[invalid-argument-type]
        )

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
            )
            self.barriers.append(collision_barrier)
            logger.info(
                "Collision avoidance enabled "
                f"with {len(self.robot.collision_model.collisionPairs)} collision pairs"
            )
        else:
            logger.info("Collision avoidance disabled")

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
    ) -> ControlResult:
        """Compute arm motion to achieve a target tool pose.

        Uses Pink's differential inverse kinematics solver with:
        1. Primary task: Achieve target pose in task space (with optional masking via costs)
        2. Secondary task (null space): Move joints toward center positions (posture task)
        3. Tertiary task: Minimize joint velocities (damping task)

        Args:
            target: Target 6-DOF pose (SE3) for the end-effector.

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

        # Set the target for the end-effector task
        self.tasks[TaskName.TOOL].set_target(target)

        # Solve inverse kinematics using Pink
        velocity = np.zeros(self.robot.model.nv)
        try:
            solved_velocity = pink.solve_ik(
                self.configuration,
                self.tasks.values(),
                self.config.dt,
                solver=self.config.solver,
                barriers=self.barriers,
                constraints=self._constraints,
            )
            velocity[self.controlled_v_indices] = solved_velocity[self.controlled_v_indices]
            self.configuration.integrate_inplace(velocity, self.config.dt)
        except Exception as e:
            # TODO: try to get unstuck if we are at limits
            logger.error(f"Encountered exception: {e}")

        return ControlResult(q=self.configuration.q.copy(), v=velocity)
