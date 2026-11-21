"""Shared types for the agent policy, control API, and operator console."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray

from humanoid.types.orchestrator import Mode

if TYPE_CHECKING:
    import pinocchio as pin


@dataclass(frozen=True)
class PendingCommand:
    command: MotionCommand
    result: Future[MotionResult]
    deadline: float | None = None


@dataclass(frozen=True, kw_only=True)
class MotionCommand:
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


@dataclass
class AgentMotion:
    duration: float
    started_at: float
    initial: pin.SE3
    target: pin.SE3
    base_target: pin.SE3 | None
    initial_gripper: NDArray[np.float64]
    gripper: NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class AgentPolicyConfig:
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


class CommandExpiredError(RuntimeError):
    """The command was not started before its receiver-local deadline."""


class CommandCancelledError(RuntimeError):
    """An operator action or caller interrupted the command."""


class AgentRoute(StrEnum):
    ROOT = "/v1"
    STATE = f"{ROOT}/state"
    CAMERA = f"{ROOT}/camera"
    COMMANDS = f"{ROOT}/commands"
    COMMAND = f"{COMMANDS}/{{command_id}}"
    CANCEL = f"{COMMAND}/cancel"

    def match_command_id(self, path: str) -> str | None:
        """Extract the single command ID segment from a matching route."""
        prefix, placeholder, suffix = self.partition("{command_id}")
        if not placeholder or not path.startswith(prefix) or not path.endswith(suffix):
            return None
        command_id = path.removeprefix(prefix).removesuffix(suffix)
        return command_id if command_id and "/" not in command_id else None


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
class AgentControlEndpoint:
    url: str


@dataclass(frozen=True, kw_only=True)
class AgentControlServerConfig:
    port: int
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


class AgentPhase(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentTask:
    instruction: str
    model: str
    mode: Mode
    session_id: str


@dataclass(frozen=True)
class AgentPanelStatus:
    phase: AgentPhase = AgentPhase.IDLE
    models: tuple[str, ...] = ()
    message: str = ""
    instruction: str = ""
    model: str = ""
    login_url: str = ""
    login_code: str = ""
