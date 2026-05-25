import lcm

from humanoid.constants import DEFAULT_LCM_URL, TOPIC_TO_TYPE, Topic
from humanoid.logger import get_logger
from humanoid.types.homing import HomingTarget
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.middleware import AcceptedTypes
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

logger = get_logger(__name__)


class Publisher:
    def __init__(self, url: str = DEFAULT_LCM_URL):
        self.lc = lcm.LCM(url)
        self.url = url

    def publish(self, data: AcceptedTypes, topic: Topic) -> None:
        expected_type = TOPIC_TO_TYPE.get(topic)
        if expected_type is not type(data):
            raise TypeError(f"Topic {topic} expects {expected_type}, but got {type(data).__name__}")

        if isinstance(data, RobotJointCommand):
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
