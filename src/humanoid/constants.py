from enum import Enum

from humanoid.types.robot import RobotCommand, RobotState

DEFAULT_LCM_URL = "udpm://239.255.76.67:7667?ttl=1"


class Topic(Enum):
    ROBOT_COMMAND = "ROBOT_COMMAND"
    ROBOT_STATE = "ROBOT_STATE"


TYPE_TO_TOPIC = {RobotCommand: Topic.ROBOT_COMMAND, RobotState: Topic.ROBOT_STATE}
TOPIC_TO_TYPE = {v: k for k, v in TYPE_TO_TOPIC.items()}
SERVO_IDS = [1]
