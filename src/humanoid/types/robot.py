from dataclasses import dataclass


@dataclass
class RobotCommand:
    timestamp: float
    joint_positions: dict[str, float]


@dataclass
class RobotState:
    timestamp: float
    joint_positions: dict[str, float]
    # TODO: add temps
    # motor_temperatures: dict[str, float]
