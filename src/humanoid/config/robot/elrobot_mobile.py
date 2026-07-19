"""Mobile Elrobot configuration."""

import numpy as np

from humanoid.hardware.actuators.config import ActuatorControlMode, ActuatorHardwareConfig
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.config import RobotHardwareConfig
from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.robot import RobotConfig, RobotName
from humanoid.types.wheels import WheelConfig, WheelType

MAIN_CONTROLLER = "main"
WHEEL_IDS = [f"wheel_{index}" for index in range(1, 4)]
JOINT_IDS = [f"arm_{index}" for index in range(1, 8)]
GRIPPER_ID = "gripper_1"

HOME_POSITION = np.array(
    [
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        -0.75,
        0.5,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]
)
REST_POSITION = np.array(
    [
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        -1.6,
        -0.1,
        1.65,
        0.0,
        0.21,
        0.0,
        2.2,
    ]
)

WHEEL_CONFIGS = [
    WheelConfig(
        frame=wheel_id,
        floor_frame=f"{wheel_id}_floor",
        radius=0.05,
        type=WheelType.OMNI,
    )
    for wheel_id in WHEEL_IDS
]
ACTUATOR_CONTROL_MODES = {
    **dict.fromkeys(WHEEL_IDS, ActuatorControlMode.VELOCITY),
    **dict.fromkeys([*JOINT_IDS, GRIPPER_ID], ActuatorControlMode.POSITION),
}
ACTUATOR_CONFIGS = {
    **{
        wheel_id: FeetechActuatorConfig(
            controller=MAIN_CONTROLLER,
            actuator_id=actuator_id,
        )
        for actuator_id, wheel_id in enumerate(WHEEL_IDS, start=250)
    },
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
OPERATIONAL_SPACE_CONFIG = OperationalSpaceConfig(
    avoid_collisions=True,
    min_collision_distance=5e-3,
    joint_centering_cost=5e-3,
    joint_centering_mask=np.array([0.0] * len(WHEEL_IDS) + [1.0] * (len(JOINT_IDS) + 1)),
    damping_cost=0.1,
    damping_mask=np.array([0.0] * len(WHEEL_IDS) + [1.0] * (len(JOINT_IDS) + 1)),
)

ELROBOT_MOBILE_CONFIG = RobotConfig(
    name=RobotName.ELROBOT_MOBILE,
    tool_frame="Gripper_Base_v1_1",
    base_frame="root_joint",
    wheels=WHEEL_CONFIGS,
    home_position=HOME_POSITION,
    rest_position=REST_POSITION,
    actuator_control_modes=ACTUATOR_CONTROL_MODES,
    hardware=HARDWARE_CONFIG,
    gripper_joint_indices=[11],
    operational_space_config=OPERATIONAL_SPACE_CONFIG,
)
