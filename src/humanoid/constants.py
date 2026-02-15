from enum import Enum

from humanoid.types.robot import RobotJointCommand, RobotState, RobotToolCommand

DEFAULT_LCM_URL = "udpm://239.255.76.67:7667?ttl=1"


class Topic(Enum):
    ROBOT_JOINT_COMMAND = "ROBOT_JOINT_COMMAND"
    ROBOT_TOOL_COMMAND = "ROBOT_TOOL_COMMAND"
    ROBOT_STATE = "ROBOT_STATE"


TYPE_TO_TOPIC = {
    RobotJointCommand: Topic.ROBOT_JOINT_COMMAND,
    RobotState: Topic.ROBOT_STATE,
    RobotToolCommand: Topic.ROBOT_TOOL_COMMAND,
}
TOPIC_TO_TYPE = {v: k for k, v in TYPE_TO_TOPIC.items()}
