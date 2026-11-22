"""Shared types for the agent policy, control API, and operator console."""

import math
import time
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, TypedDict

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.types.camera import ObservationId
from humanoid.types.observation import CameraCaptures, Observation, ObservationSubscriptions
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotName
from humanoid.types.segmentation import SegmentationResult, SegmentOperation
from humanoid.types.tools import (
    CompleteDepth,
    DepthCompletionResult,
    PerceptionOperation,
    RefineMask,
    SegmentAtPoint,
    SegmentEverything,
    SegmentInBox,
    ToolModel,
)


class EpisodeTimeExpired(Exception):
    """The task's wall-clock budget expired."""


@dataclass(frozen=True)
class TimeBudget:
    seconds: float
    started_at: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Episode time must be positive and finite.")

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def remaining(self) -> float:
        seconds = self.seconds - self.elapsed_seconds
        if seconds <= 0:
            raise EpisodeTimeExpired(f"Task time budget of {self.seconds:g} seconds expired.")
        return seconds


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


@dataclass(frozen=True)
class PendingCommand:
    command: MotionCommand
    result: Future[MotionResult]
    deadline: float | None = None


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
class AgentMotionLimits:
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
class AgentPolicyConfig(AgentMotionLimits):
    subscriptions: ObservationSubscriptions


class CommandExpiredError(RuntimeError):
    """The command was not started before its receiver-local deadline."""


class CommandCancelledError(RuntimeError):
    """An operator action or caller interrupted the command."""


class AgentRoute(StrEnum):
    ROOT = "/v1"
    STATE = f"{ROOT}/state"
    CAMERA = f"{ROOT}/camera"
    CAPTURE = f"{ROOT}/capture"
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
    capture_state_tolerance: float = 0.04


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
    robot: RobotName


@dataclass(frozen=True)
class AgentPanelStatus:
    phase: AgentPhase = AgentPhase.IDLE
    models: tuple[str, ...] = ()
    message: str = ""
    instruction: str = ""
    model: str = ""
    login_url: str = ""
    login_code: str = ""


class HarnessRobot(Protocol):
    def observation(self) -> Observation: ...
    def capture(self) -> CameraCaptures: ...
    def execute(
        self, operation: ControlOperation, arguments: Mapping[str, object], *, timeout: float
    ) -> MotionResult: ...


class ObserveAgain(ToolModel):
    pass


class MotionArguments(ToolModel):
    duration: float


class Hold(MotionArguments):
    pass


class MoveBase(MotionArguments):
    delta: tuple[float, float, float]


class MoveTool(MotionArguments):
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


class SetGripper(MotionArguments):
    positions: tuple[float, ...]


class ToolCall[Name: str, Args: ToolModel](ToolModel):
    operation: Name
    arguments: Args


type PerceptionCall = (
    ToolCall[Literal[SegmentOperation.EVERYTHING], SegmentEverything]
    | ToolCall[Literal[SegmentOperation.POINT], SegmentAtPoint]
    | ToolCall[Literal[SegmentOperation.BOX], SegmentInBox]
    | ToolCall[Literal[SegmentOperation.REFINE], RefineMask]
    | ToolCall[Literal[PerceptionOperation.COMPLETE_DEPTH], CompleteDepth]
    | ToolCall[Literal[PerceptionOperation.OBSERVE_AGAIN], ObserveAgain]
)
type MotionCall = (
    ToolCall[Literal[ControlOperation.MOVE_BASE], MoveBase]
    | ToolCall[Literal[ControlOperation.MOVE_TOOL], MoveTool]
    | ToolCall[Literal[ControlOperation.SET_GRIPPER], SetGripper]
    | ToolCall[Literal[ControlOperation.HOLD], Hold]
)


class Decision[Command](ToolModel):
    message: str
    command: Command | None


type AgentDecision = Decision[MotionCall | PerceptionCall] | Decision[PerceptionCall]


class PoseRecord(TypedDict):
    position_m: list[float]
    quaternion_xyzw: list[float]


class CameraStatus(TypedDict):
    optical_frame: str
    model: str | None
    observation_id: ObservationId
    frame_number: int
    device_timestamp: float
    timestamp_domain: str
    camera_pose: PoseRecord
    intrinsics: dict[str, object]
    registered_depth_available: bool


class ObservationStatus(TypedDict):
    cameras: dict[str, CameraStatus]
    mode: Mode
    base_pose_world: PoseRecord | None
    tool_frame: str
    tool_pose_reference_frame: str
    tool_pose: PoseRecord
    gripper_units: list[str]
    gripper_positions: list[float]
    gripper_limits: list[tuple[float, float]]
    rgb_frame: int
    observation_id: ObservationId
    rgb_size_xy: list[int]
    registered_depth_available: bool


@dataclass(frozen=True, kw_only=True)
class ToolError:
    operation: str
    error: str


@dataclass(frozen=True, kw_only=True)
class DecisionContext:
    instruction: str
    observation: ObservationStatus
    last_result: MotionResult | DepthCompletionResult | SegmentationResult | ToolError | None
    seconds_remaining: float
    limits: AgentMotionLimits
