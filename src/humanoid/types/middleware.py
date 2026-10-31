from humanoid.types.actuator import ActuatorHealthReport
from humanoid.types.camera import CameraFrame
from humanoid.types.collection import (
    CollectionEpisode,
    CollectionRequest,
    CollectionRunStatus,
    SimulationSample,
)
from humanoid.types.homing import HomingTarget
from humanoid.types.logging import LoggingRequest, LoggingStatus
from humanoid.types.node import NodeRateSample
from humanoid.types.object_state import ObjectState
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.scene import SimulationSceneState
from humanoid.types.simulation import (
    SimulationReset,
    SimulationResetStatus,
    SimulationState,
)
from humanoid.types.task_annotation import TaskAnnotation

AcceptedTypes = (
    ActuatorHealthReport
    | NodeRateSample
    | RobotJointCommand
    | RobotToolCommand
    | RobotBaseCommand
    | RobotState
    | OrchestratorMode
    | OrchestratorEvent
    | LoggingStatus
    | HomingTarget
    | SimulationSceneState
    | SimulationReset
    | SimulationResetStatus
    | SimulationState
    | LoggingRequest
    | SimulationSample
    | CollectionEpisode
    | CollectionRequest
    | CollectionRunStatus
    | CameraFrame
    | ObjectState
    | TaskAnnotation
)
