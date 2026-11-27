"""LCM recording lifecycle and ownership types."""

from dataclasses import dataclass
from enum import StrEnum


class RecordingState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass(frozen=True)
class RecordingStatus:
    timestamp: float
    state: RecordingState
    file_name: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class RecordingOwner:
    """An episode recording belongs to a robot runtime, independently of motion."""

    runtime_id: str
    episode_id: str
