"""Public configuration registry and environment-selected defaults."""

from humanoid.config.robot import ROBOT_CONFIGS
from humanoid.config.selection import ROBOT_CONFIG, ROBOT_NAME
from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.config.visualizer import VISUALIZER_CONFIG

__all__ = [
    "DEFAULT_MUJOCO_SIMULATION_CONFIG",
    "ROBOT_CONFIG",
    "ROBOT_CONFIGS",
    "ROBOT_NAME",
    "VISUALIZER_CONFIG",
]
