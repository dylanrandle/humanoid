"""Public configuration registry and environment-selected defaults."""

from humanoid.config.robot import ROBOT_CONFIGS
from humanoid.config.selection import IS_SIMULATION, ROBOT_CONFIG, ROBOT_NAME
from humanoid.config.visualizer import VISUALIZER_CONFIG

__all__ = [
    "IS_SIMULATION",
    "ROBOT_CONFIG",
    "ROBOT_CONFIGS",
    "ROBOT_NAME",
    "VISUALIZER_CONFIG",
]
