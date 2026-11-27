"""Agent tasks, executions, model decisions, and decision-loop context."""

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from humanoid.types.motion import ControlOperation, MotionLimits, MotionResult
from humanoid.types.observation import CameraCaptures, Observation, ObservationSummary
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotName
from humanoid.types.segmentation import SegmentationResult
from humanoid.types.tools import (
    DepthCompletionResult,
    DepthSamples,
    MotionCall,
    PerceptionCall,
    ToolError,
    ToolModel,
)


class EpisodeTimeExpired(Exception):
    """The task's wall-clock budget expired."""


@dataclass(frozen=True)
class TimeBudget:
    seconds: float
    started_at: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Episode time must be positive and finite.")

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def remaining(self) -> float:
        seconds = self.seconds - self.elapsed_seconds
        if seconds <= 0:
            raise EpisodeTimeExpired(f"Task time budget of {self.seconds:g} seconds expired.")
        return seconds


class AgentPhase(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentTask:
    """The work requested of an agent, independent of how it is executed."""

    instruction: str


@dataclass(frozen=True)
class AgentExecution:
    """A task bound to one model, robot, mode and admitted control context."""

    task: AgentTask
    model: str
    mode: Mode
    runtime_id: str
    robot: RobotName
    motion_session_id: str | None = None


@dataclass(frozen=True)
class AgentPanelStatus:
    phase: AgentPhase = AgentPhase.IDLE
    models: tuple[str, ...] = ()
    message: str = ""
    instruction: str = ""
    model: str = ""
    login_url: str = ""
    login_code: str = ""


class HarnessRobot(Protocol):
    def observation(self) -> Observation: ...
    def capture(self) -> CameraCaptures: ...
    def execute(
        self, operation: ControlOperation, arguments: Mapping[str, object], *, timeout: float
    ) -> MotionResult: ...


class Decision[Command](ToolModel):
    message: str
    command: Command | None


type AgentDecision = Decision[MotionCall | PerceptionCall] | Decision[PerceptionCall]


type AgentFeedback = (
    MotionResult | DepthSamples | DepthCompletionResult | SegmentationResult | ToolError
)


@dataclass(frozen=True, kw_only=True)
class DecisionContext:
    instruction: str
    observation: ObservationSummary
    last_result: AgentFeedback | None
    seconds_remaining: float
    limits: MotionLimits
