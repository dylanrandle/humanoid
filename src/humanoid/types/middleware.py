from humanoid.types.actuator import ActuatorHealthReport
from humanoid.types.camera import CameraImage, RegisteredDepth
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
from humanoid.types.simulation import SimulationState

AcceptedTypes = (
    SimulationState
    | ActuatorHealthReport
    | RegisteredDepth
    | CameraImage
    | NodeRateSample
    | RobotJointCommand
    | RobotToolCommand
    | RobotBaseCommand
    | RobotState
    | OrchestratorMode
    | OrchestratorEvent
    | LoggingStatus
    | HomingTarget
)
