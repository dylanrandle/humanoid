from dataclasses import dataclass, field
from typing import Literal, TypedDict, get_args

from humanoid.constants import Topic
from humanoid.types.camera import CameraFrames, CameraSummary, CameraView, PoseRecord
from humanoid.types.homing import HomingTarget
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotBaseCommand, RobotJointCommand, RobotState, RobotToolCommand

ObservationTopic = Literal[
    Topic.ROBOT_STATE,
    Topic.ROBOT_JOINT_COMMAND,
    Topic.ROBOT_TOOL_COMMAND,
    Topic.ROBOT_BASE_COMMAND,
    Topic.ORCHESTRATOR_MODE,
    Topic.HOMING_TARGET,
]


@dataclass(frozen=True, kw_only=True)
class ObservationSubscriptions:
    """Feedback topics and named RGB-D cameras to include in an observation."""

    topics: tuple[ObservationTopic, ...]
    camera_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if Topic.ROBOT_STATE not in self.topics:
            raise ValueError("Every observation requires ROBOT_STATE.")
        if set(self.topics).difference(get_args(ObservationTopic)):
            raise ValueError("Unsupported observation topic.")
        if len(set(self.topics)) != len(self.topics):
            raise ValueError("Duplicate observation topic.")
        if len({name.upper() for name in self.camera_names}) != len(self.camera_names):
            raise ValueError("Duplicate camera name.")


@dataclass
class Observation:
    robot_state: RobotState
    robot_joint_command: RobotJointCommand | None = None
    robot_tool_command: RobotToolCommand | None = None
    robot_base_command: RobotBaseCommand | None = None
    mode: Mode | None = None
    camera_frames: dict[str, CameraFrames] = field(default_factory=dict)
    homing_target: HomingTarget | None = None


type CameraCaptures = dict[str, tuple[Observation, CameraView]]


@dataclass(frozen=True, kw_only=True)
class ObservationConfig:
    feedback_timeout: float = 0.25
    camera_timeout_seconds: float = 0.5
    capture_state_tolerance: float = 0.04


type CameraFrameCaptures = dict[str, tuple[Observation, CameraFrames]]


class ObservationSummary(TypedDict):
    """Measured state for tools and trial programs; primary_camera is a cameras key."""

    cameras: dict[str, CameraSummary]
    mode: Mode
    base_pose_world: PoseRecord | None
    tool_frame: str
    tool_pose_reference_frame: str
    tool_pose: PoseRecord
    gripper_units: list[str]
    gripper_positions: list[float]
    gripper_limits: list[tuple[float, float]]
    primary_camera: str
