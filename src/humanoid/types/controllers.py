from dataclasses import dataclass

import numpy as np


@dataclass
class OperationalSpaceConfig:
    """Configuration parameters for the operational space controller."""

    tool_position_cost: float = 1.0
    tool_orientation_cost: float = 1.0
    base_position_cost: float = 1.0
    base_orientation_cost: float = 1.0
    wheel_cost: float = 1.0
    dt: float = 0.005
    max_linear_velocity: float = 1.0
    max_angular_velocity: float = np.pi
    joint_centering_cost: float = 1e-3
    damping_cost: float = 1e-1
    solver: str = "quadprog"
    avoid_collisions: bool = False
    joint_centering_mask: np.ndarray | float = 1.0
    damping_mask: np.ndarray | float = 1.0
