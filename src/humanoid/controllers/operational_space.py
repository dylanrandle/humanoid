"""Operational space controller using Pink for 6-DOF task space control.

This controller computes joint commands to achieve target task space poses using
the Pink inverse kinematics library.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pink
import pinocchio as pin
from numpy.typing import NDArray
from pink.barriers import SelfCollisionBarrier
from pink.tasks import DampingTask, FrameTask, OmniwheelTask, PostureTask, RollingTask
from pink.utils import process_collision_pairs

from humanoid.logger import get_logger
from humanoid.robots.base import Robot
from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.robot import WheelType

logger = get_logger(__name__)


class TaskName(StrEnum):
    """Enum for task names used in the operational space controller."""

    TOOL = "tool"
    BASE = "base"
    JOINT_CENTERING = "joint_centering"
    DAMPING = "damping"


@dataclass
class ControlResult:
    """Result of a single control step."""

    q: NDArray[np.float64]  # joint configuration (nq,)
    v: NDArray[np.float64]  # joint velocity (nv,)


class OperationalSpaceController:
    """Operational space controller for 6-DOF task space control using Pink."""

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

        # Defer configuration initialization until first state update
        self.configuration: pink.Configuration | None = None

        # Create tasks dictionary
        self.tasks = {}

        # Create end-effector frame task
        robot.assert_frame_exists(robot.config.tool_frame)
        self.tasks[TaskName.TOOL] = FrameTask(
            robot.config.tool_frame,
            position_cost=self.config.tool_position_cost,
            orientation_cost=self.config.tool_orientation_cost,
        )

        # Create base frame task if base_frame is configured
        if robot.config.base_frame is not None:
            robot.assert_frame_exists(robot.config.base_frame)
            self.tasks[TaskName.BASE] = FrameTask(
                robot.config.base_frame,
                position_cost=self.config.base_position_cost,
                orientation_cost=self.config.base_orientation_cost,
            )

            # Add wheels if defined
            for wheel in robot.config.wheels or []:
                robot.assert_frame_exists(wheel.frame)
                robot.assert_frame_exists(wheel.floor_frame)
                task_cls = OmniwheelTask if wheel.type is WheelType.OMNI else RollingTask
                self.tasks[wheel.frame] = task_cls(
                    wheel.frame,
                    floor_frame=wheel.floor_frame,
                    wheel_radius=wheel.radius,
                    cost=self.config.wheel_cost,
                )

        # Create posture task for null space control (joint centering)
        self.tasks[TaskName.JOINT_CENTERING] = PostureTask(
            cost=self.config.joint_centering_cost * self.config.joint_centering_mask  # ty:ignore[invalid-argument-type]
        )
        self.tasks[TaskName.JOINT_CENTERING].set_target(robot.config.home_position)

        # Create damping task for velocity minimization
        self.tasks[TaskName.DAMPING] = DampingTask(
            cost=self.config.damping_cost * self.config.damping_mask  # ty:ignore[invalid-argument-type]
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

    def update_state(self, q: np.ndarray):
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
                self.robot.data,
                q,
                collision_model=collision_model,
                collision_data=collision_data,
            )
            logger.debug(f"Initialized controller configuration with state: q={q}")
        else:
            self.configuration.update(q)

    def compute_control(
        self,
        tool_target_pose: pin.SE3,
        base_target_pose: pin.SE3 | None = None,
        gripper_positions: NDArray[np.float64] | None = None,
    ) -> ControlResult:
        """Compute joint configuration to achieve target task space pose.

        Uses Pink's differential inverse kinematics solver with:
        1. Primary task: Achieve target pose in task space (with optional masking via costs)
        2. Secondary task (null space): Move joints toward center positions (posture task)
        3. Tertiary task: Minimize joint velocities (damping task)

        Args:
            tool_target_pose: Target 6-DOF pose (SE3) for the end-effector.
            base_target_pose: Optional target 6-DOF pose (SE3) for the base frame.
                Only used if the robot has a base_frame configured.
            gripper_positions: Optional gripper joint positions to override in the result

        Returns:
            ControlResult with q (nq,) and v (nv,); q has gripper positions overridden if provided

        Raises:
            RuntimeError: If configuration has not been initialized via update_state()
        """
        if self.configuration is None:
            raise RuntimeError(
                "Controller configuration not initialized. "
                "Call update_state() with robot state first."
            )

        # Set the target for the end-effector task
        self.tasks[TaskName.TOOL].set_target(tool_target_pose)

        # Set the target for the base task if provided and configured
        if TaskName.BASE in self.tasks and base_target_pose is not None:
            self.tasks[TaskName.BASE].set_target(base_target_pose)

        # Solve inverse kinematics using Pink
        velocity = np.zeros(self.robot.model.nv)
        try:
            velocity = pink.solve_ik(
                self.configuration,
                self.tasks.values(),
                self.config.dt,
                solver=self.config.solver,
                barriers=self.barriers,
            )
            self.configuration.integrate_inplace(velocity, self.config.dt)
        except Exception as e:
            # TODO: try to get unstuck if we are at limits
            logger.error(f"Encountered exception: {e}")

        # Get the computed joint configuration
        q = self.configuration.q.copy()

        # Override gripper joint positions if provided
        # TODO: consider other ways of doing this
        if gripper_positions is not None:
            self.robot.set_gripper_positions(q, gripper_positions)

        return ControlResult(q=q, v=velocity)
