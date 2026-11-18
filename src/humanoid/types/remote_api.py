"""Transport contracts for the standalone robot-control service."""

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import StrEnum

from humanoid.types.remote_control import MotionResult


class ControlOperation(StrEnum):
    MOVE_TOOL = "move_tool"
    MOVE_BASE = "move_base"
    SET_GRIPPER = "set_gripper"
    HOLD = "hold"


class CommandState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass(frozen=True)
class CommandStatus:
    command_id: str
    state: CommandState
    result: MotionResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class CommandRecord:
    fingerprint: str
    result: Future[MotionResult]


@dataclass(frozen=True)
class RemoteControlEndpoint:
    url: str
    token: str = field(repr=False)


@dataclass(frozen=True, kw_only=True)
class RemoteControlServerConfig:
    host: str
    port: int
    token: str = field(repr=False)
    max_commands: int = 4096
    max_request_bytes: int = 65536
    max_start_delay_seconds: float = 2.0
    camera_timeout_seconds: float = 0.5


@dataclass(frozen=True)
class CommandRequest:
    command_id: str
    operation: ControlOperation
    arguments: Mapping[str, object]
    expires_in_seconds: float
