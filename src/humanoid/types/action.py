from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class Action:
    joint_positions: np.ndarray | None = None
    tool_pose: pin.SE3 | None = None
    gripper_positions: np.ndarray | None = None
    base_pose: pin.SE3 | None = None

    # TODO: consider moving this into policies themselves (more explicit)
    def __post_init__(self) -> None:
        if self.tool_pose is not None and self.base_pose is not None:
            self.tool_pose = self.base_pose * self.tool_pose
