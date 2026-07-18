import numpy as np

from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotConfig, RobotName
from humanoid.types.servo import ServoControlMode
from humanoid.types.visualizer import VisualizerConfig
from humanoid.types.wheels import WheelConfig, WheelType

IS_SIMULATION = Runtime.from_environment() is Runtime.SIM
ROBOT_NAME = RobotName.from_environment()

ROBOT_CONFIGS = {
    cfg.name: cfg
    for cfg in [
        RobotConfig(
            name=RobotName.PANDA,
            tool_frame="panda_hand_tcp",
            home_position=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]),
            rest_position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.785, 0.0]),
            joint_idx_to_servo_id={i: i for i in range(8)},
            servo_control_modes=dict.fromkeys(range(8), ServoControlMode.POSITION),
            gripper_joint_indices=[7],
            operational_space_config=OperationalSpaceConfig(avoid_collisions=True),
        ),
        RobotConfig(
            name=RobotName.SO101,
            tool_frame="gripper_frame_link",
            home_position=np.array([0.0, -0.5, 0.8, -0.3, 0.0, 0.0]),
            rest_position=np.array([0.0, -1.55, 1.5, 1.0, 0.0, -0.15]),
            # NOTE: servos 5 <-> 6 are swapped
            joint_idx_to_servo_id={0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 5},
            servo_control_modes=dict.fromkeys(range(1, 7), ServoControlMode.POSITION),
            gripper_joint_indices=[5],
        ),
        RobotConfig(
            name=RobotName.ELROBOT,
            tool_frame="Gripper_Base_v1_1",
            home_position=np.array([0.0, -0.75, 0.5, 0.0, 0.0, 1.0, 0.0, 0]),
            rest_position=np.array([0.0, -1.6, -0.1, 1.65, 0.0, 0.21, 0.0, 2.2]),
            joint_idx_to_servo_id={i: i + 1 for i in range(8)},
            servo_control_modes=dict.fromkeys(range(1, 9), ServoControlMode.POSITION),
            # NOTE: servo 8 direction is inverted
            inverted_servo_ids=[8],
            gripper_joint_indices=[7],
        ),
        RobotConfig(
            name=RobotName.ELROBOT_MOBILE,
            tool_frame="Gripper_Base_v1_1",
            base_frame="root_joint",
            wheels=[
                WheelConfig(
                    frame=f"wheel_{i}",
                    floor_frame=f"wheel_{i}_floor",
                    radius=0.05,
                    type=WheelType.OMNI,
                )
                for i in (1, 2, 3)
            ],
            home_position=np.array(
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
                    0,
                ]
            ),
            rest_position=np.array(
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
            ),
            joint_idx_to_servo_id={
                1: 250,
                2: 251,
                3: 252,
                4: 1,
                5: 2,
                6: 3,
                7: 4,
                8: 5,
                9: 6,
                10: 7,
                11: 8,
            },
            servo_control_modes={
                **dict.fromkeys([250, 251, 252], ServoControlMode.VELOCITY),
                **dict.fromkeys(range(1, 9), ServoControlMode.POSITION),
            },
            # NOTE: servo 8 direction is inverted
            inverted_servo_ids=[8],
            gripper_joint_indices=[11],
            operational_space_config=OperationalSpaceConfig(
                avoid_collisions=True,
                min_collision_distance=5e-3,
                joint_centering_cost=5e-3,
                joint_centering_mask=np.array([0.0] * 3 + [1.0] * 8),
                damping_cost=0.1,
                damping_mask=np.array([0.0] * 3 + [1.0] * 8),
            ),
        ),
    ]
}

ROBOT_CONFIG = ROBOT_CONFIGS[ROBOT_NAME]

VISUALIZER_CONFIG = VisualizerConfig()
