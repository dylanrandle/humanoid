from dataclasses import dataclass

from humanoid.types.robot import RobotBaseCommand, RobotJointCommand, RobotState, RobotToolCommand


@dataclass
class Observation:
    robot_state: RobotState
    robot_joint_command: RobotJointCommand | None = None
    robot_tool_command: RobotToolCommand | None = None
    robot_base_command: RobotBaseCommand | None = None
