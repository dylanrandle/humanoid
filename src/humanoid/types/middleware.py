from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

AcceptedTypes = RobotJointCommand | RobotToolCommand | RobotBaseCommand | RobotState
