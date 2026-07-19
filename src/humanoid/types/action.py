from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class Action:
    """Policy output using the selected robot's configured tool-command frame."""

    joint_positions: np.ndarray | None = None
    tool_pose: pin.SE3 | None = None
    gripper_positions: np.ndarray | None = None
    base_pose: pin.SE3 | None = None
