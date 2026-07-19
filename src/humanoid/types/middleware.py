from humanoid.types.homing import HomingTarget
from humanoid.types.logging import LoggingStatus
from humanoid.types.node import NodeRateSample
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

AcceptedTypes = (
    NodeRateSample
    | RobotJointCommand
    | RobotToolCommand
    | RobotBaseCommand
    | RobotState
    | OrchestratorMode
    | OrchestratorEvent
    | LoggingStatus
    | HomingTarget
)
