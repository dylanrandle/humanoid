import math
from dataclasses import dataclass

from humanoid.types.actuator import ActuatorHealthReport
from humanoid.types.camera import CameraImage, RegisteredDepth
from humanoid.types.homing import HomingTarget
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
    | HomingTarget
)


@dataclass(frozen=True, kw_only=True)
class PipeConfig:
    """Bounds and worker timing for a typed JSON pipe."""

    capacity: int
    max_bytes: int
    poll_seconds: float
    join_seconds: float

    def __post_init__(self) -> None:
        for name in ("capacity", "max_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Pipe {name} must be a positive integer.")
        for name in ("poll_seconds", "join_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Pipe {name} must be positive and finite.")
