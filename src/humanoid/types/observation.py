from dataclasses import dataclass

from humanoid.types.robot import RobotState


@dataclass
class Observation:
    robot_state: RobotState
