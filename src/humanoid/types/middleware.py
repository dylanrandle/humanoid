from humanoid.types.homing import HomingTarget
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

AcceptedTypes = (
    RobotJointCommand
    | RobotToolCommand
    | RobotBaseCommand
    | RobotState
    | OrchestratorMode
    | OrchestratorEvent
    | HomingTarget
)
