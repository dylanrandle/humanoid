"""SO-101 robot configuration."""

import numpy as np

from humanoid.hardware.actuators.config import ActuatorControlMode, ActuatorHardwareConfig
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.config import RobotHardwareConfig
from humanoid.types.robot import RobotConfig, RobotName

MAIN_CONTROLLER = "main"
JOINT_IDS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# Actuators 5 and 6 are intentionally swapped on this robot.
ACTUATOR_IDS = [1, 2, 3, 4, 6, 5]

HOME_POSITION = np.array([0.0, -0.5, 0.8, -0.3, 0.0, 0.0])
REST_POSITION = np.array([0.0, -1.55, 1.5, 1.0, 0.0, -0.15])

ACTUATOR_CONTROL_MODES = dict.fromkeys(JOINT_IDS, ActuatorControlMode.POSITION)
ACTUATOR_CONFIGS = {
    joint_id: FeetechActuatorConfig(
        controller=MAIN_CONTROLLER,
        actuator_id=actuator_id,
    )
    for joint_id, actuator_id in zip(JOINT_IDS, ACTUATOR_IDS, strict=True)
}
HARDWARE_CONFIG = RobotHardwareConfig(
    actuators=ActuatorHardwareConfig(
        controllers={MAIN_CONTROLLER: FeetechActuatorControllerConfig()},
        joints=ACTUATOR_CONFIGS,
    ),
)

SO101_CONFIG = RobotConfig(
    name=RobotName.SO101,
    tool_frame="gripper_frame_link",
    home_position=HOME_POSITION,
    rest_position=REST_POSITION,
    actuator_control_modes=ACTUATOR_CONTROL_MODES,
    hardware=HARDWARE_CONFIG,
    gripper_joint_indices=[5],
)
