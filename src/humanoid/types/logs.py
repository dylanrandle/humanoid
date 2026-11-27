"""Log entries and cursor-based snapshots."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LogEntry:
    cursor: int
    message: str


@dataclass(frozen=True)
class LogSnapshot:
    cursor: int
    entries: list[LogEntry]
    reset: bool
    capacity: int
