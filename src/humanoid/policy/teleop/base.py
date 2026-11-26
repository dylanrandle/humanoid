"""Shared base class for teleoperation policies.

Holds the logic that's common to keyboard, VR, or any other teleop input:
loading the robot for forward kinematics, validating and looking up the
gripper joint, and turning the latest observation into a "hold-current-pose"
Action used by uninitialized / dead-man / disengaged paths.
"""

import numpy as np
import pinocchio as pin

from humanoid.config import ROBOT_CONFIG
from humanoid.logger import get_logger
from humanoid.policy.base import Policy
from humanoid.robots.base import Robot
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)


class BaseTeleopPolicy(Policy):
    """Base class for teleop policies that command tool/base/gripper targets.

    Subclasses implement ``step`` (and optionally ``reset``) for their
    specific input device. This class owns:

    - the :class:`Robot` instance used for forward kinematics,
    - the single-gripper-joint precondition and ``(min, max)`` limits, and
    - helpers for building hold-current-pose actions from an observation,
      including correct position-index access into the observation's
      ``joint_positions`` array.
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        verbose: bool = True,
    ):
        """Initialize shared teleop state.

        Args:
            robot_config: Robot configuration used to construct the FK model.
            verbose: Whether to emit init-time log lines.
        """
        self.robot_config = robot_config
        self.verbose = verbose
        self.robot = Robot(robot_config)

        gripper = robot_config.gripper
        if gripper is not None:
            assert len(gripper.joint_names) == 1, (
                f"{type(self).__name__} only supports 1 gripper joint, "
                f"but {len(gripper.joint_names)} were specified"
            )
            (self.gripper_min, self.gripper_max), *_ = self.robot.get_gripper_limits()
            if self.verbose:
                logger.info(
                    f"Gripper joint {gripper.joint_names[0]}: "
                    f"[{self.gripper_min:.4f}, {self.gripper_max:.4f}] {gripper.units[0]}"
                )
        else:
            self.gripper_min = 0.0
            self.gripper_max = 0.0

    def _step_gripper(self, position: float, step: float, *, close: bool) -> float:
        """Move a single gripper joint toward its physical open or closed limit."""
        gripper = self.robot_config.gripper
        assert gripper is not None
        delta = step if close == gripper.closes_at_upper_limit else -step
        return float(np.clip(position + delta, self.gripper_min, self.gripper_max))

    def _get_current_tool_pose(self, observation: Observation) -> pin.SE3:
        """Tool pose suitable for use as ``Action.tool_pose``.

        When a mobile base is configured, returns the tool in that base frame
        for the controller's relative-frame task. Returns world-frame otherwise.
        """
        # Pinocchio can return a view into its reusable frame-placement buffer.
        q = self.robot.hold_configuration(observation.robot_state, observation.robot_joint_command)
        return self.robot.get_tool_command_pose(q).copy()

    def _get_current_base_pose(self, observation: Observation) -> pin.SE3 | None:
        """Base pose from FK on joint positions.

        Returns None when no mobile base is configured.
        """
        pose = self.robot.get_base_pose(self._base_reference_positions(observation))
        return pose.copy() if pose is not None else None

    def _get_current_gripper_positions(self, observation: Observation) -> np.ndarray | None:
        """Read current gripper positions from the observation.

        Uses the joint→position index mapping so this stays correct even when
        a floating base joint shifts the configuration layout. Returns None
        when no gripper joints are configured.
        """
        position_indices = self.robot.get_gripper_position_indices()
        if not position_indices:
            return None
        q = self.robot.hold_configuration(observation.robot_state, observation.robot_joint_command)
        return q[position_indices]

    def _hold_current_pose_action(self, observation: Observation) -> Action:
        """Build an Action that asks the controller to hold the current state.

        Returns the FK tool pose, FK base pose (when configured), and the
        current gripper positions — i.e. "stay where you are."
        """
        return Action(
            tool_pose=self._get_current_tool_pose(observation),
            gripper_positions=self._get_current_gripper_positions(observation),
            base_pose=self._get_current_base_pose(observation),
        )
