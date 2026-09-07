"""Robot configuration registry."""

from humanoid.config.robot.panda import PANDA_CONFIG
from humanoid.config.robot.so101 import SO101_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.types.robot import RobotConfig

ROBOT_CONFIGS: dict[str, RobotConfig] = {
    config.name: config
    for config in (
        PANDA_CONFIG,
        SO101_CONFIG,
        TRISKEL_CONFIG,
    )
}

__all__ = [
    "PANDA_CONFIG",
    "ROBOT_CONFIGS",
    "SO101_CONFIG",
    "TRISKEL_CONFIG",
]
