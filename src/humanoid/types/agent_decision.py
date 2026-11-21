"""Model decisions, available command variants, and feedback for the agent loop."""

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from humanoid.types.agent import AgentPolicyConfig, ControlOperation, MotionResult
from humanoid.types.camera import ObservationId
from humanoid.types.orchestrator import Mode
from humanoid.types.segmentation import SegmentationResult, SegmentOperation
from humanoid.types.tools import (
    CompleteDepth,
    DepthCompletionResult,
    RefineMask,
    SegmentAtPoint,
    SegmentEverything,
    SegmentInBox,
    ToolModel,
)


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
    | ToolCall[Literal["complete_depth"], CompleteDepth]
    | ToolCall[Literal["observe_again"], ObserveAgain]
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


type AgentDecision = (
    Decision[MotionCall | PerceptionCall]
    | Decision[MotionCall]
    | Decision[PerceptionCall]
    | Decision[None]
)


class PoseRecord(TypedDict):
    position_m: list[float]
    quaternion_xyzw: list[float]


class ObservationStatus(TypedDict):
    mode: Mode
    base_pose_world: PoseRecord
    tool_frame: str
    tool_pose_base: PoseRecord
    gripper_positions_rad: list[float]
    gripper_limits_rad: list[tuple[float, float]]
    rgb_frame: int
    observation_id: ObservationId
    rgb_size_xy: list[int]
    registered_depth_available: bool
    measured_depth_preview_range_m: NotRequired[list[float]]


@dataclass(frozen=True, kw_only=True)
class ToolError:
    operation: str
    error: str


@dataclass(frozen=True, kw_only=True)
class DecisionContext:
    instruction: str
    observation: ObservationStatus
    last_result: MotionResult | DepthCompletionResult | SegmentationResult | ToolError | None
    commands_remaining: int
    perception_remaining: int
    limits: AgentPolicyConfig
