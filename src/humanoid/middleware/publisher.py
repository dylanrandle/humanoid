import lcm

from humanoid.constants import DEFAULT_LCM_URL, Channel, message_type
from humanoid.logger import get_logger
from humanoid.types.actuator import ActuatorHealthReport
from humanoid.types.camera import (
    CameraFrames,
    CameraImage,
    CameraTopic,
    CameraTopics,
    RegisteredDepth,
)
from humanoid.types.homing import HomingTarget
from humanoid.types.lcm import registered_depth_t
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.middleware import AcceptedTypes
from humanoid.types.node import NodeRateSample
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.simulation import SimulationState

logger = get_logger(__name__)


class Publisher:
    def __init__(self, url: str = DEFAULT_LCM_URL):
        self.lc = lcm.LCM(url)
        self.url = url

    def publish_camera(self, frames: CameraFrames, topics: CameraTopics) -> None:
        if frames.registered is not None:
            self.publish(frames.registered, topics.registered_depth)
        self.publish(frames.rgb, topics.rgb)
        self.publish(frames.depth, topics.depth)

    def publish(self, data: AcceptedTypes, topic: Channel) -> None:  # noqa: PLR0912 - typed dispatch
        expected_type = message_type(topic)
        if expected_type is not type(data):
            raise TypeError(f"Topic {topic} expects {expected_type}, but got {type(data).__name__}")

        if isinstance(data, CameraImage):
            if not isinstance(topic, CameraTopic) or data.encoding is not topic.encoding:
                raise ValueError(f"Topic {topic.value} requires its configured image encoding.")
            lcm_data = LCMConverter.camera_image_to_lcm(data)
        elif isinstance(data, SimulationState):
            lcm_data = LCMConverter.simulation_state_to_lcm(data)
        elif isinstance(data, RegisteredDepth):
            lcm_data = registered_depth_t()
            lcm_data.camera_model = data.camera_model.value
            lcm_data.rgb_frame_number = data.rgb_frame_number
            lcm_data.image = LCMConverter.camera_image_to_lcm(data.image)
        elif isinstance(data, ActuatorHealthReport):
            lcm_data = LCMConverter.actuator_health_report_to_lcm(data)
        elif isinstance(data, NodeRateSample):
            lcm_data = LCMConverter.node_rate_sample_to_lcm(data)
        elif isinstance(data, RobotJointCommand):
            lcm_data = LCMConverter.robot_joint_command_to_lcm(data)
        elif isinstance(data, RobotState):
            lcm_data = LCMConverter.robot_state_to_lcm(data)
        elif isinstance(data, RobotToolCommand):
            lcm_data = LCMConverter.robot_tool_command_to_lcm(data)
        elif isinstance(data, RobotBaseCommand):
            lcm_data = LCMConverter.robot_base_command_to_lcm(data)
        elif isinstance(data, OrchestratorMode):
            lcm_data = LCMConverter.orchestrator_mode_to_lcm(data)
        elif isinstance(data, OrchestratorEvent):
            lcm_data = LCMConverter.orchestrator_event_to_lcm(data)
        elif isinstance(data, HomingTarget):
            lcm_data = LCMConverter.homing_target_to_lcm(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.lc.publish(topic.value, lcm_data.encode())
