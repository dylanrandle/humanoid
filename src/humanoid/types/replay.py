"""Types for LCM log replay."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from humanoid.types.robot import RobotName

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True)
class RecordingBundle:
    """Validated paths and metadata for one server-managed recording."""

    id: str
    directory: Path
    log_path: Path
    manifest_path: Path
    robot: RobotName
    created_at: str
    robot_config: dict[str, JsonValue]

    @property
    def file_name(self) -> str:
        return str(self.log_path)


@dataclass(frozen=True)
class RecordingSummary:
    """Recording metadata exposed to the operator console."""

    id: str
    robot: RobotName
    created_at: str


class ReplayOutcome(StrEnum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class ReplayStatus:
    """Current or most recent replay state."""

    running: bool
    file_name: str | None
    outcome: ReplayOutcome | None
    exit_code: int | None
    last_output: str | None
