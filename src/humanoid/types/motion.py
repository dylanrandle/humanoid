"""Typed motion requests, execution state, transport and policy configuration."""

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Protocol, TypedDict

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from humanoid.types.observation import ObservationSubscriptions


@dataclass(frozen=True, kw_only=True)
class MotionTarget:
    """Optional tool/base/gripper targets over a bounded duration.

    An empty command holds the current pose. Tool poses use the robot's tool
    command frame; base_delta is [x, y, yaw] relative to the base at execution
    start. Omitted gripper targets preserve the existing squeeze target.
    """

    duration: float = 1.0
    tool_pose: pin.SE3 | None = None
    base_delta: NDArray[np.float64] | None = None
    gripper_positions: NDArray[np.float64] | None = None


class MotionResult(TypedDict):
    """Measured motion outcome; completion does not imply reaching the target."""

    reached: bool
    position_error_m: float
    orientation_error_rad: float
    gripper_error_rad: float
    elapsed_seconds: float
    step_id: int


@dataclass(frozen=True)
class PendingMotion:
    command: MotionTarget
    result: Future[MotionResult]
    deadline: float | None = None


@dataclass
class ActiveMotion:
    duration: float
    started_at: float
    initial: pin.SE3
    target: pin.SE3
    base_target: pin.SE3 | None
    initial_gripper: NDArray[np.float64]
    gripper: NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class MotionLimits:
    """Policy timing, bounded command horizons, and feedback requirements.

    Tool/base speeds come from RobotConfig, and joint bounds come from the URDF.
    Gripper closing time sets the rate of traversing that joint range, as in teleop.
    The workspace bounds constrain the tool origin, not collision clearance.
    """

    dt: float = 0.02
    feedback_timeout: float = 0.25
    max_duration: float = 2.0
    max_tool_translation: float = 0.08
    max_tool_rotation: float = 0.5
    max_base_translation: float = 0.15
    max_base_rotation: float = 0.5
    gripper_close_time: float = 3.0
    tool_workspace_min: tuple[float, float, float] = (-0.35, -0.35, 0.01)
    tool_workspace_max: tuple[float, float, float] = (0.35, 0.45, 0.48)
    position_tolerance: float = 0.008
    orientation_tolerance: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "dt",
            "feedback_timeout",
            "max_duration",
            "max_tool_translation",
            "max_tool_rotation",
            "max_base_translation",
            "max_base_rotation",
            "gripper_close_time",
            "position_tolerance",
            "orientation_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for value in (self.tool_workspace_min, self.tool_workspace_max):
            if np.shape(value) != (3,) or not np.isfinite(value).all():
                raise ValueError("Tool workspace must use finite 3D coordinates.")
        if np.any(np.asarray(self.tool_workspace_min) >= self.tool_workspace_max):
            raise ValueError("Tool workspace minimum must be below maximum.")


@dataclass(frozen=True, kw_only=True)
class MotionPolicyConfig(MotionLimits):
    subscriptions: ObservationSubscriptions


class MotionExpiredError(RuntimeError):
    """The command was not started before its receiver-local deadline."""


class MotionCancelledError(RuntimeError):
    """An operator action or caller interrupted the command."""


class ControlOperation(StrEnum):
    MOVE_TOOL = "move_tool"
    MOVE_BASE = "move_base"
    SET_GRIPPER = "set_gripper"
    HOLD = "hold"


class MotionState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass(frozen=True)
class MotionStatus:
    motion_id: str
    state: MotionState
    result: MotionResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class MotionRecord:
    fingerprint: str
    result: Future[MotionResult]


@dataclass(frozen=True, kw_only=True)
class MotionServiceConfig:
    max_motions: int = 4096
    max_start_delay_seconds: float = 2.0


@dataclass(frozen=True, kw_only=True)
class MotionProtocolConfig:
    """Liveness and acknowledgement timing for the motion request protocol."""

    heartbeat_seconds: float
    contact_timeout_seconds: float
    startup_timeout_seconds: float
    request_timeout_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "heartbeat_seconds",
            "contact_timeout_seconds",
            "startup_timeout_seconds",
            "request_timeout_seconds",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Motion protocol {name} must be positive and finite.")


@dataclass(frozen=True)
class MotionRequest:
    motion_id: str
    operation: ControlOperation
    arguments: Mapping[str, object]
    expires_in_seconds: float


class MotionNotFound(KeyError):
    """The requested motion is unknown in this session."""


class MotionSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    motion_session_id: str
    motion_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    operation: ControlOperation
    arguments: dict[str, object]
    expires_in_seconds: float = Field(gt=0, strict=True)


class MotionSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    motion_session_id: str


class MotionMessageKind(StrEnum):
    SUBMIT = "submit"
    STATUS = "status"
    CANCEL = "cancel"
    PING = "ping"
    REPLY = "reply"


class MotionErrorKind(StrEnum):
    INVALID = "invalid"
    MISSING = "missing"
    CONFLICT = "conflict"


class MotionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: MotionMessageKind
    motion_session_id: str
    correlation_id: str
    motion_id: str = ""
    request: MotionRequest | None = None
    deadline: float | None = Field(default=None, ge=0)
    status: MotionStatus | None = None
    ready: bool = False
    error: str | None = None
    error_kind: MotionErrorKind = MotionErrorKind.CONFLICT


class MotionProcessLifecycle(Protocol):
    """Process manager boundary for a motion transport owned by the backend."""

    def attach(self, connection: Connection, motion_session_id: str) -> None: ...
    def detach(self) -> None: ...


class MotionArguments(BaseModel):
    """Shared API/tool arguments; runtime safety limits are checked by the policy."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    duration: float = 1.0

    @field_validator(
        "delta", "position", "quaternion", "positions", mode="before", check_fields=False
    )
    @classmethod
    def accept_json_vectors(cls, value: object) -> object:
        # JSON arrays and Python tuples share the same strict numeric validation.
        return tuple(value) if isinstance(value, list) else value


class Hold(MotionArguments):
    pass


class MoveBase(MotionArguments):
    delta: tuple[float, float, float]


class MoveTool(MotionArguments):
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]

    @field_validator("quaternion")
    @classmethod
    def unit_quaternion(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if not np.isclose(np.linalg.norm(value), 1.0, atol=1e-5):
            raise ValueError("quaternion must have unit norm, in x,y,z,w order.")
        return value


class SetGripper(MotionArguments):
    positions: tuple[float, ...]
