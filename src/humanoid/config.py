import os

import numpy as np

from humanoid.types.robot import RobotConfig
from humanoid.types.servo import ServoControlMode
from humanoid.types.visualizer import VisualizerConfig

IS_SIMULATION = os.getenv("HUMANOID_RUNTIME", "sim").lower().strip() in ("sim", "simulation")
ROBOT_NAME = os.getenv("HUMANOID_ROBOT", "elrobot_mobile").lower().strip()

ROBOT_CONFIGS = {
    cfg.name: cfg
    for cfg in [
        RobotConfig(
            name="panda",
            tool_frame="panda_hand_tcp",
            home_position=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]),
            rest_position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.785, 0.0]),
            joint_idx_to_servo_id={i: i for i in range(8)},
            servo_control_modes=dict.fromkeys(range(8), ServoControlMode.POSITION),
            gripper_joint_indices=[7],
        ),
        RobotConfig(
            name="so101",
            tool_frame="gripper_frame_link",
            home_position=np.array([0.0, -0.5, 0.8, -0.3, 0.0, 0.0]),
            rest_position=np.array([0.0, -1.55, 1.5, 1.0, 0.0, -0.15]),
            # NOTE: servos 5 <-> 6 are swapped
            joint_idx_to_servo_id={0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 5},
            servo_control_modes=dict.fromkeys(range(1, 7), ServoControlMode.POSITION),
            gripper_joint_indices=[5],
        ),
        RobotConfig(
            name="elrobot",
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
            name="elrobot_mobile",
            tool_frame="Gripper_Base_v1_1",
            base_frame="base_link",
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
                0: 250,
                1: 251,
                2: 252,
                3: 1,
                4: 2,
                5: 3,
                6: 4,
                7: 5,
                8: 6,
                9: 7,
                10: 8,
            },
            servo_control_modes={
                **dict.fromkeys([250, 251, 252], ServoControlMode.VELOCITY),
                **dict.fromkeys(range(1, 9), ServoControlMode.POSITION),
            },
            # NOTE: servo 8 direction is inverted
            inverted_servo_ids=[8],
            gripper_joint_indices=[10],
        ),
    ]
}

assert ROBOT_NAME in ROBOT_CONFIGS, (
    f"{ROBOT_NAME} not recognized, available options: {list(ROBOT_CONFIGS.keys())}"
)
ROBOT_CONFIG = ROBOT_CONFIGS[ROBOT_NAME]

VISUALIZER_CONFIG = VisualizerConfig()
