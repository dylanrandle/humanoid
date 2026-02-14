import os
from enum import Enum

import numpy as np

from humanoid.types.robot import RobotConfig, RobotJointCommand, RobotState, RobotToolCommand

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
ROBOT_NAME = "panda"
ROBOT_CONFIGS = {
    cfg.name: cfg
    for cfg in [
        RobotConfig(
            name="panda",
            end_effector_frame="panda_hand_tcp",
            home_position=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]),
            servo_ids=list(range(9)),
        ),
        RobotConfig(
            name="so101",
            end_effector_frame="tcp",
            home_position=np.array([0.0, 0.0, 0.0, 0.0]),
            servo_ids=list(range(4)),
        ),
    ]
}
assert ROBOT_NAME in ROBOT_CONFIGS, (
    f"{ROBOT_NAME} not recognized, available options: {list(ROBOT_CONFIGS.keys())}"
)
ROBOT_CONFIG = ROBOT_CONFIGS[ROBOT_NAME]
IS_SIMULATION = os.getenv("RUNTIME_ENV", "sim").lower() in ("sim", "simulation")
