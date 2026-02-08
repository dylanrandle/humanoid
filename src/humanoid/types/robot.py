from dataclasses import dataclass


@dataclass
class RobotCommand:
    timestamp: float
    joint_positions: dict[str, float]


@dataclass
class RobotState:
    timestamp: float
    joint_positions: dict[str, float]
    motor_temperatures: dict[str, float]
