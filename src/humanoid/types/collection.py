"""Shared episode contracts and explicitly privileged simulation payloads."""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pinocchio as pin

from humanoid.types.camera import CameraFrame
from humanoid.types.object_state import ObjectState
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotJointCommand, RobotName, RobotState
from humanoid.types.task_annotation import TaskAnnotation


class EpisodeOutcome(StrEnum):
    READY = "ready"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class TaskResult:
    success: bool
    reason: str


@dataclass(frozen=True)
class CollectionConfig:
    control_hz: int = 50
    sample_hz: int = 10
    max_sim_seconds: float = 600.0
    startup_timeout: float = 5.0
    finalization_timeout: float = 5.0

    def __post_init__(self) -> None:
        if self.control_hz <= 0 or self.sample_hz <= 0:
            raise ValueError("Policy and sample rates must be positive.")
        if (
            not np.isfinite(
                [self.max_sim_seconds, self.startup_timeout, self.finalization_timeout]
            ).all()
            or min(self.max_sim_seconds, self.startup_timeout, self.finalization_timeout) <= 0
        ):
            raise ValueError("Collection time limits must be positive and finite.")


@dataclass(frozen=True)
class SimulationCameraObservation:
    """A reusable camera observation plus privileged simulation extrinsics."""

    frame: CameraFrame
    world_from_optical: pin.SE3


@dataclass(frozen=True)
class SimulationSample:
    """One timestamped simulator observation with its last applied command.

    The stack runs asynchronously on host_monotonic. State, images, objects,
    and labels share the snapshot timestamp; command retains its application
    time. action_index is -1 because it does not index a synchronized trajectory.
    """

    timestamp: float
    episode_id: str
    sequence: int
    action_index: int
    state: RobotState
    command: RobotJointCommand
    cameras: tuple[SimulationCameraObservation, ...]
    objects: tuple[ObjectState, ...]
    task_annotation: TaskAnnotation


@dataclass(frozen=True)
class CollectionEpisode:
    """Episode lifecycle for simulation experts and real teleoperation alike.

    timestamp is in seconds on the named clock shared by observation/action
    producers. Counts are cumulative; -1 means unavailable or not applicable
    for asynchronous streams. metadata_json carries runtime-specific settings,
    provenance and outcome evidence; a simulation seed is not required here.
    """

    timestamp: float
    episode_id: str
    task: str
    task_version: int
    runtime: Runtime
    clock: str
    outcome: EpisodeOutcome
    sample_count: int
    action_count: int
    metadata_json: str


@dataclass(frozen=True)
class CollectionReport:
    directory: Path
    outcome: EpisodeOutcome
    valid: bool
    reason: str
    sample_count: int
    action_count: int
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def eligible_for_training(self) -> bool:
        return self.outcome is EpisodeOutcome.SUCCEEDED and self.valid


class CollectionRunState(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class CollectionAction(StrEnum):
    PREPARE = "prepare"
    START = "start"
    STOP = "stop"


class CollectionCancelled(Exception):
    """The operator stopped the current collection session."""


@dataclass(frozen=True)
class CollectionTaskOption:
    name: str
    label: str
    robot: RobotName | None
    collectable: bool = True


@dataclass(frozen=True)
class CollectionSettings:
    task: str
    episodes: int
    max_attempts: int
    seed: int
    max_sim_seconds: float
    output: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task, str)
            or not self.task.strip()
            or not isinstance(self.output, str)
            or not self.output.strip()
            or "\x00" in self.output
        ):
            raise ValueError("Select a task and recording directory.")
        if any(type(value) is not int for value in (self.episodes, self.max_attempts, self.seed)):
            raise ValueError("Episodes, attempts, and seed must be integers.")
        if self.episodes <= 0 or self.max_attempts < self.episodes or self.seed < 0:
            raise ValueError("Require positive episodes, attempts >= episodes, and seed >= 0.")
        if self.seed + self.max_attempts >= 2**53 or self.max_attempts >= 2**31:
            raise ValueError("Attempts or seed exceed supported integer bounds.")
        if (
            type(self.max_sim_seconds) not in (int, float)
            or not math.isfinite(self.max_sim_seconds)
            or self.max_sim_seconds <= 0
        ):
            raise ValueError("Episode time limit must be positive and finite.")


@dataclass(frozen=True)
class CollectionRequest:
    timestamp: float
    run_id: str
    action: CollectionAction
    settings: CollectionSettings | None = None


@dataclass(frozen=True)
class CollectionRunStatus:
    timestamp: float
    state: CollectionRunState = CollectionRunState.IDLE
    run_id: str = ""
    task: str = ""
    episodes: int = 0
    max_attempts: int = 0
    attempts: int = 0
    successes: int = 0
    seed: int = 0
    phase: str = ""
    target: str = ""
    sim_seconds: float = 0.0
    actions: int = 0
    samples: int = 0
    recording: str = ""
    message: str = ""

    @property
    def running(self) -> bool:
        return self.state in {
            CollectionRunState.PREPARING,
            CollectionRunState.STARTING,
            CollectionRunState.RUNNING,
            CollectionRunState.STOPPING,
        }
