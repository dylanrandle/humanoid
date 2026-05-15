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

    Subclasses implement ``__call__`` (and optionally ``reset``) for their
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

        if robot_config.gripper_joint_indices:
            assert len(robot_config.gripper_joint_indices) == 1, (
                f"{type(self).__name__} only supports 1 gripper joint, "
                f"but {len(robot_config.gripper_joint_indices)} were specified"
            )
            (self.gripper_min, self.gripper_max), *_ = self.robot.get_gripper_limits()
            if self.verbose:
                gripper_joint_idx = robot_config.gripper_joint_indices[0]
                gripper_range = self.gripper_max - self.gripper_min
                logger.info(
                    f"Gripper joint {gripper_joint_idx}: "
                    f"[{self.gripper_min:.4f}, {self.gripper_max:.4f}] "
                    f"(range: {gripper_range:.4f}, {gripper_range * 1000:.2f}mm)"
                )
        else:
            self.gripper_min = 0.0
            self.gripper_max = 0.0

    def _get_current_tool_pose(self, observation: Observation) -> pin.SE3:
        """Forward-kinematics tool pose at the observation's joint positions."""
        return self.robot.get_tool_pose(observation.robot_state.joint_positions)

    def _get_current_base_pose(self, observation: Observation) -> pin.SE3 | None:
        """Forward-kinematics base pose, or None when no base_frame is configured."""
        return self.robot.get_base_pose(observation.robot_state.joint_positions)

    def _get_current_gripper_positions(self, observation: Observation) -> np.ndarray | None:
        """Read current gripper positions from the observation.

        Uses the joint→position index mapping so this stays correct even when
        a floating base joint shifts the configuration layout. Returns None
        when no gripper joints are configured.
        """
        position_indices = self.robot.get_gripper_position_indices()
        if not position_indices:
            return None
        joint_positions = observation.robot_state.joint_positions
        return np.array([joint_positions[idx] for idx in position_indices])

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
