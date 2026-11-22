"""Triskel robot configuration."""

from dataclasses import replace

import numpy as np

from humanoid.config.actuator import (
    STS3215_12V_CURRENT_CALIBRATION,
    STS3215_12V_EFFORT_LIMITS,
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
from humanoid.types.actuator import ActuatorControlMode, ActuatorHardwareConfig
from humanoid.types.camera import RobotCameraConfig
from humanoid.types.controllers import OmniwheelBaseConfig, OperationalSpaceConfig
from humanoid.types.depth import DepthCamera
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    CartesianVelocityLimits,
    MeshContactConfig,
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
MAX_POSITION_VELOCITY_RAD_S = 2.0
POSITION_PID_GAINS_BY_ACTUATOR_ID: dict[int, FeetechPIDGains] = {
    1: FeetechPIDGains(p=16, i=0, d=32),
    2: FeetechPIDGains(p=32, i=1, d=32),
    3: FeetechPIDGains(p=32, i=1, d=32),
    4: FeetechPIDGains(p=32, i=1, d=32),
    5: FeetechPIDGains(p=16, i=0, d=32),
    6: FeetechPIDGains(p=16, i=0, d=32),
    7: FeetechPIDGains(p=16, i=0, d=32),
    8: FeetechPIDGains(p=16, i=0, d=32),
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
            max_position_velocity=MAX_POSITION_VELOCITY_RAD_S,
            position_pid=POSITION_PID_GAINS_BY_ACTUATOR_ID[actuator_id],
            effort_limits=STS3215_12V_EFFORT_LIMITS,
            current_calibration=STS3215_12V_CURRENT_CALIBRATION,
        )
        for actuator_id, joint_id in enumerate(JOINT_IDS, start=1)
    },
    GRIPPER_ID: FeetechActuatorConfig(
        controller=MAIN_CONTROLLER,
        actuator_id=8,
        inverted=True,
        max_position_velocity=MAX_POSITION_VELOCITY_RAD_S,
        position_pid=POSITION_PID_GAINS_BY_ACTUATOR_ID[8],
        effort_limits=STS3215_12V_EFFORT_LIMITS,
        current_calibration=STS3215_12V_CURRENT_CALIBRATION,
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
GRIPPER_CONFIG = RobotGripperConfig(joint_names=(GRIPPER_ID,), units=("rad",))
# A simplified hull keeps the near-coplanar fin facets stable against flat toys.
# The short contact response limits penetration under the rack-driven jaws.
FIN_CONTACT = MeshContactConfig(
    max_hull_vertices=16,
    priority=1,
    solref=(0.01, 1),
    solimp=(0.95, 0.99, 0.001, 0.5, 2),
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
    actuator_control_modes=ACTUATOR_CONTROL_MODES,
    hardware=HARDWARE_CONFIG,
    state_estimation=STATE_ESTIMATION_CONFIG,
    gripper=GRIPPER_CONFIG,
    mesh_contacts={"finger_1_fin_contact": FIN_CONTACT, "finger_2_fin_contact": FIN_CONTACT},
    operational_space_config=OPERATIONAL_SPACE_CONFIG,
    omniwheel_base_config=OMNIWHEEL_BASE_CONFIG,
)
