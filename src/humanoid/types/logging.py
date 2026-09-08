"""Data-logging lifecycle types."""

from dataclasses import dataclass
from enum import StrEnum


class LoggingState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass(frozen=True)
class LoggingStatus:
    timestamp: float
    state: LoggingState
    file_name: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ApplicationLogEntry:
    cursor: int
    message: str


@dataclass(frozen=True)
class ApplicationLogSnapshot:
    cursor: int
    entries: list[ApplicationLogEntry]
    reset: bool
    capacity: int
