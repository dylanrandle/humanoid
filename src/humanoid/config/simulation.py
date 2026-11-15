"""Native MuJoCo simulation configuration defaults."""

from humanoid.config.camera import SIMULATED_CAMERA_CONFIG
from humanoid.types.simulation import MujocoSimulationConfig

DEFAULT_MUJOCO_SIMULATION_CONFIG = MujocoSimulationConfig(camera=SIMULATED_CAMERA_CONFIG)
