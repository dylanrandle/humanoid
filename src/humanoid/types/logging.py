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
