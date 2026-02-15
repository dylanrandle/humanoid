import os

import numpy as np

from humanoid.types.robot import RobotConfig
from humanoid.types.visualizer import VisualizerConfig

IS_SIMULATION = os.getenv("HUMANOID_RUNTIME", "sim").lower().strip() in ("sim", "simulation")
ROBOT_NAME = os.getenv("HUMANOID_ROBOT", "panda").lower().strip()

ROBOT_CONFIGS = {
    cfg.name: cfg
    for cfg in [
        RobotConfig(
            name="panda",
            end_effector_frame="panda_hand_tcp",
            home_position=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]),
            joint_idx_to_servo_id={i: i for i in range(9)},
        ),
        RobotConfig(
            name="so101",
            end_effector_frame="tcp",
            home_position=np.array([0.0, 0.0, 0.0, 0.0]),
            joint_idx_to_servo_id={i: i + 1 for i in range(4)},
        ),
    ]
}

assert ROBOT_NAME in ROBOT_CONFIGS, (
    f"{ROBOT_NAME} not recognized, available options: {list(ROBOT_CONFIGS.keys())}"
)
ROBOT_CONFIG = ROBOT_CONFIGS[ROBOT_NAME]

VISUALIZER_CONFIG = VisualizerConfig()
