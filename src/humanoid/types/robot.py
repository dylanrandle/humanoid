from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class RobotJointCommand:
    timestamp: float
    joint_positions: dict[str, float]


@dataclass
class RobotState:
    timestamp: float
    joint_positions: dict[str, float]
    motor_temperatures: dict[str, float]


@dataclass
class RobotToolCommand:
    timestamp: float
    pose: pin.SE3


@dataclass
class RobotConfig:
    name: str
    end_effector_frame: str
    home_position: np.ndarray
    servo_ids: list[int]
