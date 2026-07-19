"""Panda robot configuration."""

import numpy as np

from humanoid.hardware.actuators.config import ActuatorControlMode
from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.robot import RobotConfig, RobotName

JOINT_IDS = [
    *[f"panda_joint{index}" for index in range(1, 8)],
    "panda_finger_joint1",
]

HOME_POSITION = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04])
REST_POSITION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.785, 0.0])

ACTUATOR_CONTROL_MODES = dict.fromkeys(JOINT_IDS, ActuatorControlMode.POSITION)
OPERATIONAL_SPACE_CONFIG = OperationalSpaceConfig(avoid_collisions=True)

PANDA_CONFIG = RobotConfig(
    name=RobotName.PANDA,
    tool_frame="panda_hand_tcp",
    home_position=HOME_POSITION,
    rest_position=REST_POSITION,
    actuator_control_modes=ACTUATOR_CONTROL_MODES,
    hardware=None,
    gripper_joint_indices=[7],
    operational_space_config=OPERATIONAL_SPACE_CONFIG,
)
