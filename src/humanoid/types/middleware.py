from humanoid.types.orchestrator import OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

AcceptedTypes = (
    RobotJointCommand | RobotToolCommand | RobotBaseCommand | RobotState | OrchestratorMode
)
