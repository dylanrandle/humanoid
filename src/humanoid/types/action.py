from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class Action:
    """Action type that can represent either joint space or tool space commands.

    Exactly one of joint_positions or tool_pose should be set.
    Optionally, gripper_positions can be specified alongside tool_pose, and
    base_pose can be specified alongside either to command the robot's base.
    """

    joint_positions: np.ndarray | None = None
    tool_pose: pin.SE3 | None = None
    gripper_positions: np.ndarray | None = None
    base_pose: pin.SE3 | None = None

    def __post_init__(self) -> None:
        """Validate that exactly one action type is specified."""
        if self.joint_positions is None and self.tool_pose is None:
            raise ValueError("Either joint_positions or tool_pose must be specified")
        if self.joint_positions is not None and self.tool_pose is not None:
            raise ValueError("Only one of joint_positions or tool_pose can be specified")

    @property
    def is_joint_action(self) -> bool:
        """Check if this is a joint space action."""
        return self.joint_positions is not None

    @property
    def is_tool_action(self) -> bool:
        """Check if this is a tool space action."""
        return self.tool_pose is not None
