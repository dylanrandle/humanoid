from dataclasses import dataclass

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
