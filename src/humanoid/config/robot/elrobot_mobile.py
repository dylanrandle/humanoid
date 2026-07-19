"""Mobile Elrobot configuration."""

import numpy as np

from humanoid.hardware.actuators.config import ActuatorControlMode, ActuatorHardwareConfig
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.config import RobotHardwareConfig
from humanoid.state_estimation.config import RobotStateEstimationConfig
from humanoid.state_estimation.root.wheel_dead_reckoning import (
    WheelDeadReckoningRootStateEstimatorConfig,
)
from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    CartesianVelocityLimits,
    RobotBaseConfig,
    RobotConfig,
    RobotName,
    RobotToolConfig,
    WheelConfig,
    WheelType,
)

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
STATE_ESTIMATION_CONFIG = RobotStateEstimationConfig(
    root=WheelDeadReckoningRootStateEstimatorConfig(),
)
TOOL_CONFIG = RobotToolConfig(frame="Gripper_Base_v1_1")
BASE_CONFIG = RobotBaseConfig(
    frame="root_joint",
    velocity_limits=CartesianVelocityLimits(
        linear=0.2,
        angular=1.0,
    ),
)
HOMING_PRESETS = {
    HomingPreset.HOME: HOME_POSITION,
    HomingPreset.REST: REST_POSITION,
}
OPERATIONAL_SPACE_CONFIG = OperationalSpaceConfig(
    avoid_collisions=True,
    wheel_cost=100.0,
    min_collision_distance=5e-3,
    joint_centering_cost=5e-3,
    joint_centering_mask=np.array([0.0] * len(WHEEL_IDS) + [1.0] * (len(JOINT_IDS) + 1)),
    damping_cost=0.1,
    damping_mask=np.array([0.0] * len(WHEEL_IDS) + [1.0] * (len(JOINT_IDS) + 1)),
)

ELROBOT_MOBILE_CONFIG = RobotConfig(
    name=RobotName.ELROBOT_MOBILE,
    tool=TOOL_CONFIG,
    base=BASE_CONFIG,
    wheels=WHEEL_CONFIGS,
    homing_presets=HOMING_PRESETS,
    actuator_control_modes=ACTUATOR_CONTROL_MODES,
    hardware=HARDWARE_CONFIG,
    state_estimation=STATE_ESTIMATION_CONFIG,
    gripper_joint_indices=[11],
    operational_space_config=OPERATIONAL_SPACE_CONFIG,
)
