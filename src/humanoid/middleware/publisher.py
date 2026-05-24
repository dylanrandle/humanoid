import lcm

from humanoid.constants import DEFAULT_LCM_URL, TYPE_TO_TOPIC
from humanoid.logger import get_logger
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.middleware import AcceptedTypes
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

    def publish(self, data: AcceptedTypes) -> None:
        # Convert to LCM type based on data type
        if isinstance(data, RobotJointCommand):
            lcm_data = LCMConverter.robot_joint_command_to_lcm(data)
            topic = TYPE_TO_TOPIC[RobotJointCommand]
        elif isinstance(data, RobotState):
            lcm_data = LCMConverter.robot_state_to_lcm(data)
            topic = TYPE_TO_TOPIC[RobotState]
        elif isinstance(data, RobotToolCommand):
            lcm_data = LCMConverter.robot_tool_command_to_lcm(data)
            topic = TYPE_TO_TOPIC[RobotToolCommand]
        elif isinstance(data, RobotBaseCommand):
            lcm_data = LCMConverter.robot_base_command_to_lcm(data)
            topic = TYPE_TO_TOPIC[RobotBaseCommand]
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Encode and publish
        data_bytes = lcm_data.encode()
        self.lc.publish(topic.value, data_bytes)
