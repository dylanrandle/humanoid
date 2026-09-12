"""Triskel robot configuration."""

import numpy as np

from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
    FeetechPIDGains,
)
from humanoid.hardware.config import RobotHardwareConfig
from humanoid.state_estimation.config import RobotStateEstimationConfig
from humanoid.state_estimation.root.wheel_dead_reckoning import (
    WheelDeadReckoningRootStateEstimatorConfig,
)
from humanoid.types.actuator import ActuatorControlMode, ActuatorHardwareConfig
from humanoid.types.controllers import OmniwheelBaseConfig, OperationalSpaceConfig
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    CartesianVelocityLimits,
    RobotBaseConfig,
    RobotConfig,
    RobotGripperConfig,
    RobotName,
    RobotToolConfig,
    WheelConfig,
    WheelType,
)

MAIN_CONTROLLER = "main"
CONTROLLER_RATE_HZ = 10.0
CONTROLLER_DT = 1 / CONTROLLER_RATE_HZ
WHEEL_IDS = [f"wheel_{index}" for index in range(1, 4)]
JOINT_IDS = [f"arm_{index}" for index in range(1, 8)]
GRIPPER_ID = "gripper_1"
POSITION_PID_GAINS_BY_ACTUATOR_ID: dict[int, FeetechPIDGains] = {
    1: FeetechPIDGains(p=32, i=0, d=32),
    2: FeetechPIDGains(p=32, i=0, d=32),
    3: FeetechPIDGains(p=32, i=0, d=32),
    4: FeetechPIDGains(p=32, i=0, d=32),
    5: FeetechPIDGains(p=32, i=0, d=32),
    6: FeetechPIDGains(p=32, i=0, d=32),
    7: FeetechPIDGains(p=32, i=0, d=32),
    8: FeetechPIDGains(p=32, i=0, d=32),
}

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
            position_pid=POSITION_PID_GAINS_BY_ACTUATOR_ID[actuator_id],
        )
        for actuator_id, joint_id in enumerate(JOINT_IDS, start=1)
    },
    GRIPPER_ID: FeetechActuatorConfig(
        controller=MAIN_CONTROLLER,
        actuator_id=8,
        inverted=True,
        position_pid=POSITION_PID_GAINS_BY_ACTUATOR_ID[8],
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
TOOL_CONFIG = RobotToolConfig(frame="gripper_base_link")
GRIPPER_CONFIG = RobotGripperConfig(joint_names=(GRIPPER_ID,))
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
    dt=CONTROLLER_DT,
    tool_position_cost=5.0,
    avoid_collisions=True,
    min_collision_distance=5e-3,
    collision_safe_displacement_gain=1e-2,
    joint_centering_cost=5e-3,
    damping_cost=0.1,
    low_acceleration_cost=1e-2,
    joint_velocity_limit=1.0,
    joint_acceleration_limit=2.0,
)
OMNIWHEEL_BASE_CONFIG = OmniwheelBaseConfig(dt=CONTROLLER_DT)
TRISKEL_CONFIG = RobotConfig(
    name=RobotName.TRISKEL,
    tool=TOOL_CONFIG,
    base=BASE_CONFIG,
    wheels=WHEEL_CONFIGS,
    homing_presets=HOMING_PRESETS,
    actuator_control_modes=ACTUATOR_CONTROL_MODES,
    hardware=HARDWARE_CONFIG,
    state_estimation=STATE_ESTIMATION_CONFIG,
    gripper=GRIPPER_CONFIG,
    operational_space_config=OPERATIONAL_SPACE_CONFIG,
    omniwheel_base_config=OMNIWHEEL_BASE_CONFIG,
)
