"""Elrobot arm configuration."""

import numpy as np

from humanoid.hardware.actuators.config import ActuatorControlMode, ActuatorHardwareConfig
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.config import RobotHardwareConfig
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    RobotConfig,
    RobotName,
    RobotToolConfig,
)

MAIN_CONTROLLER = "main"
JOINT_IDS = [f"arm_{index}" for index in range(1, 8)]
GRIPPER_ID = "gripper_1"

HOME_POSITION = np.array([0.0, -0.75, 0.5, 0.0, 0.0, 1.0, 0.0, 0])
REST_POSITION = np.array([0.0, -1.6, -0.1, 1.65, 0.0, 0.21, 0.0, 2.2])
TOOL_CONFIG = RobotToolConfig(frame="Gripper_Base_v1_1")
HOMING_PRESETS = {
    HomingPreset.HOME: HOME_POSITION,
    HomingPreset.REST: REST_POSITION,
}

ACTUATOR_CONTROL_MODES = dict.fromkeys([*JOINT_IDS, GRIPPER_ID], ActuatorControlMode.POSITION)
ACTUATOR_CONFIGS = {
    **{
        joint_id: FeetechActuatorConfig(
            controller=MAIN_CONTROLLER,
            actuator_id=actuator_id,
        )
        for actuator_id, joint_id in enumerate(JOINT_IDS, start=1)
    },
    GRIPPER_ID: FeetechActuatorConfig(
        controller=MAIN_CONTROLLER,
        actuator_id=8,
        inverted=True,
    ),
}
HARDWARE_CONFIG = RobotHardwareConfig(
    actuators=ActuatorHardwareConfig(
        controllers={MAIN_CONTROLLER: FeetechActuatorControllerConfig()},
        joints=ACTUATOR_CONFIGS,
    ),
)

ELROBOT_CONFIG = RobotConfig(
    name=RobotName.ELROBOT,
    tool=TOOL_CONFIG,
    homing_presets=HOMING_PRESETS,
    actuator_control_modes=ACTUATOR_CONTROL_MODES,
    hardware=HARDWARE_CONFIG,
    gripper_joint_indices=[7],
)
