"""Triskel robot configuration."""

from dataclasses import replace

import numpy as np

from humanoid.config.actuator import (
    STS3215_7V4_CURRENT_CALIBRATION,
    STS3215_7V4_RATED_TORQUE,
    STS3215_345_OUTPUT_ARMATURE,
)
from humanoid.config.camera import (
    REALSENSE_CAMERA_CONFIG,
    SIMULATED_CAMERA_CONFIG,
    WRIST_D405_CONFIG,
)
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
from humanoid.types.actuator import ActuatorControlMode, ActuatorHardwareConfig, JointDriveConfig
from humanoid.types.camera import RobotCameraConfig
from humanoid.types.controllers import OmniwheelBaseConfig, OperationalSpaceConfig
from humanoid.types.depth import DepthCamera
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    CartesianVelocityLimits,
    GeomContactConfig,
    RobotBaseConfig,
    RobotConfig,
    RobotGripperConfig,
    RobotName,
    RobotToolConfig,
    WheelConfig,
    WheelType,
)

MAIN_CONTROLLER = "main"
CONTROLLER_RATE_HZ = 30.0
CONTROLLER_DT = 1 / CONTROLLER_RATE_HZ
WHEEL_IDS = [f"wheel_{index}" for index in range(1, 4)]
JOINT_IDS = [f"arm_{index}" for index in range(1, 8)]
GRIPPER_ID = "gripper_1"
GRIPPER_ACTUATOR_ID = 8
MAX_POSITION_VELOCITY_RAD_S = 2.0
POSITION_PID_GAINS_BY_ACTUATOR_ID: dict[int, FeetechPIDGains] = {
    1: FeetechPIDGains(p=16, i=0, d=32),
    2: FeetechPIDGains(p=32, i=1, d=32),
    3: FeetechPIDGains(p=32, i=1, d=32),
    4: FeetechPIDGains(p=32, i=1, d=32),
    5: FeetechPIDGains(p=16, i=0, d=32),
    6: FeetechPIDGains(p=16, i=0, d=32),
    7: FeetechPIDGains(p=16, i=0, d=32),
    GRIPPER_ACTUATOR_ID: FeetechPIDGains(p=16, i=0, d=32),
}


HOME_POSITION = np.array(
    [
        # floating base
        0.0,
        0.0,
        1.0,
        0.0,
        # wheel 1
        1.0,
        0.0,
        # wheel 2
        1.0,
        0.0,
        # wheel 3
        1.0,
        0.0,
        # j1
        0.0,
        # j2
        -0.82,
        # j3
        0.82,
        # j4
        0.0,
        # j5
        0.0,
        # j6
        0.83,
        # j7
        0.0,
        # gripper
        0.0,
    ]
)
REST_POSITION = np.array(
    [
        # floating base
        0.0,
        0.0,
        1.0,
        0.0,
        # wheel 1
        1.0,
        0.0,
        # wheel 2
        1.0,
        0.0,
        # wheel 3
        1.0,
        0.0,
        # j1
        0.0,
        # j2
        -1.57,
        # j3
        1.57,
        # j4
        0.0,
        # j5
        0.0,
        # j6
        1.57,
        # j7
        0.0,
        # gripper
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
JOINT_DRIVES = {
    **dict.fromkeys(
        WHEEL_IDS,
        JointDriveConfig(
            control_mode=ActuatorControlMode.VELOCITY, armature=STS3215_345_OUTPUT_ARMATURE
        ),
    ),
    **dict.fromkeys(
        [*JOINT_IDS, GRIPPER_ID],
        JointDriveConfig(
            control_mode=ActuatorControlMode.POSITION, armature=STS3215_345_OUTPUT_ARMATURE
        ),
    ),
}
ACTUATOR_CONFIGS = {
    **{
        wheel_id: FeetechActuatorConfig(
            controller=MAIN_CONTROLLER,
            actuator_id=actuator_id,
            rated_effort=STS3215_7V4_RATED_TORQUE,
            current_calibration=STS3215_7V4_CURRENT_CALIBRATION,
        )
        for actuator_id, wheel_id in enumerate(WHEEL_IDS, start=250)
    },
    **{
        joint_id: FeetechActuatorConfig(
            controller=MAIN_CONTROLLER,
            actuator_id=actuator_id,
            max_position_velocity=MAX_POSITION_VELOCITY_RAD_S,
            position_pid=POSITION_PID_GAINS_BY_ACTUATOR_ID[actuator_id],
            rated_effort=STS3215_7V4_RATED_TORQUE,
            current_calibration=STS3215_7V4_CURRENT_CALIBRATION,
        )
        for actuator_id, joint_id in enumerate(JOINT_IDS, start=1)
    },
    GRIPPER_ID: FeetechActuatorConfig(
        controller=MAIN_CONTROLLER,
        actuator_id=GRIPPER_ACTUATOR_ID,
        inverted=True,
        max_position_velocity=MAX_POSITION_VELOCITY_RAD_S,
        position_pid=POSITION_PID_GAINS_BY_ACTUATOR_ID[GRIPPER_ACTUATOR_ID],
        rated_effort=STS3215_7V4_RATED_TORQUE,
        current_calibration=STS3215_7V4_CURRENT_CALIBRATION,
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
GRIPPER_CONFIG = RobotGripperConfig(
    joint_names=(GRIPPER_ID,),
    units=("rad",),
    closes_at_upper_limit=True,
)
# Fin Ray body friction from TRI's LBM Eval 1.1.0 mesh-mimic hand. This is a
# reference value until measured against Triskel's actual toy surfaces.
TPU_PAD_CONTACT = GeomContactConfig(
    priority=1,
    solref=(0.0025, 1),
    sliding_friction=3.0,
    torsional_friction=0.005,
    solimp=(0.995, 0.9999, 0.001, 0.5, 2),
)
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
    tool_orientation_cost=0.72,
    avoid_collisions=True,
    min_collision_distance=5e-3,
    collision_safe_displacement_gain=1e-2,
    manipulability_cost=0.03,
    damping_cost=0.3,
    low_acceleration_cost=0.0,
    joint_velocity_limit=1.0,
    joint_acceleration_limit=2.0,
    joint_position_margin=5e-3,
)
OMNIWHEEL_BASE_CONFIG = OmniwheelBaseConfig(dt=CONTROLLER_DT)
TRISKEL_CONFIG = RobotConfig(
    cameras={
        "base": RobotCameraConfig(
            simulation=SIMULATED_CAMERA_CONFIG,
            hardware=replace(
                REALSENSE_CAMERA_CONFIG,
                serial_number="242522071106",
                expected_model=DepthCamera.D435,
            ),
        ),
        "wrist": RobotCameraConfig(
            simulation=WRIST_D405_CONFIG,
            hardware=replace(
                REALSENSE_CAMERA_CONFIG,
                serial_number="130322270287",
                expected_model=DepthCamera.D405,
                rgb_frame=WRIST_D405_CONFIG.rgb_frame,
                depth_frame=WRIST_D405_CONFIG.depth_frame,
            ),
        ),
    },
    name=RobotName.TRISKEL,
    tool=TOOL_CONFIG,
    base=BASE_CONFIG,
    wheels=WHEEL_CONFIGS,
    homing_presets=HOMING_PRESETS,
    joint_drives=JOINT_DRIVES,
    hardware=HARDWARE_CONFIG,
    state_estimation=STATE_ESTIMATION_CONFIG,
    gripper=GRIPPER_CONFIG,
    geom_contacts={
        "finger_1_fin_contact": TPU_PAD_CONTACT,
        "finger_2_fin_contact": TPU_PAD_CONTACT,
    },
    operational_space_config=OPERATIONAL_SPACE_CONFIG,
    omniwheel_base_config=OMNIWHEEL_BASE_CONFIG,
)
