from dataclasses import dataclass, field
from typing import Literal, get_args

from humanoid.constants import Topic
from humanoid.types.camera import CameraFrames, CameraView
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
