from dataclasses import dataclass, field

from humanoid.types.camera import CameraFrames, CameraView
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotBaseCommand, RobotJointCommand, RobotState, RobotToolCommand


@dataclass
class Observation:
    robot_state: RobotState
    robot_joint_command: RobotJointCommand | None = None
    robot_tool_command: RobotToolCommand | None = None
    robot_base_command: RobotBaseCommand | None = None
    mode: Mode | None = None
    camera_frames: dict[str, CameraFrames] = field(default_factory=dict)


type CameraCaptures = dict[str, tuple[Observation, CameraView]]
