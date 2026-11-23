"""Controller submission, feedback and evaluation records for simulated agent trials."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

import mujoco
from pydantic import JsonValue

from humanoid.types.agent import (
    AgentDecision,
    AgentFeedback,
    DecisionContext,
    MotionCall,
    MotionResult,
    ObservationStatus,
)
from humanoid.types.simulation import SimulationScene
from humanoid.types.tools import ToolModel


class TrialSubmission(ToolModel):
    source: str


class TrialStep(ToolModel):
    command: MotionCall | None
    memory: dict[str, object]


class TrialMotionError(TypedDict):
    error: str


type TrialFeedback = MotionResult | TrialMotionError


class ControllerImage(TypedDict):
    rgb: list[list[list[int]]]
    registered_depth_m: list[list[float]]
    raw_depth_m: list[list[float]]


class TrialObservation(ObservationStatus):
    seconds_remaining: float
    images: dict[str, ControllerImage]
    last_result: TrialFeedback | None


class TrialTraceEntry(TypedDict):
    command: dict[str, object]
    result: TrialFeedback


@dataclass
class ExplorationTurn:
    elapsed_seconds: float
    context: DecisionContext
    images: dict[str, str]
    decision: AgentDecision | None = None
    result: AgentFeedback | None = None
    result_image: str | None = None


@dataclass(frozen=True)
class TaskScore:
    success: bool
    metrics: dict[str, JsonValue]


class SimulationTask(ABC):
    """A task selects a scene, configures each reset and judges simulator state."""

    scene: SimulationScene
    completion_reason: str = "task_complete"

    @property
    @abstractmethod
    def goal(self) -> str: ...

    def configure_scene(self, spec: mujoco.MjSpec, seed: int) -> None:  # noqa: B027
        """Optionally modify the composed spec before an episode is compiled."""

    @abstractmethod
    def score(self, data: mujoco.MjData) -> TaskScore: ...


class EpisodeEnd(StrEnum):
    COMPLETE = "task_complete"
    TIME_BUDGET = "time_budget"
    AGENT = "agent_finished"
    CONTROLLER = "controller_finished"
    ERROR = "controller_error"


@dataclass(frozen=True, kw_only=True)
class TrialResult:
    seed: int
    elapsed_seconds: float
    trace: list[TrialTraceEntry]
    score: TaskScore
    ended_by: str
    error: str | None
    lcm_recording: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.score.success

    def record(self) -> dict[str, object]:
        return {
            **self.score.metrics,
            **({"lcm_recording": self.lcm_recording} if self.lcm_recording else {}),
            "seed": self.seed,
            "trace": self.trace,
            "elapsed_seconds": self.elapsed_seconds,
            "ended_by": self.ended_by,
            "success": self.success,
            "error": self.error,
        }
