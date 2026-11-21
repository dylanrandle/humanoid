"""Native MuJoCo simulation configuration defaults."""

import os

from humanoid.config.camera import SIMULATED_CAMERA_CONFIG
from humanoid.constants import SCENE_ENVIRONMENT_VARIABLE
from humanoid.types.simulation import MujocoSimulationConfig, SimulationScene

DEFAULT_MUJOCO_SIMULATION_CONFIG = MujocoSimulationConfig(
    camera=SIMULATED_CAMERA_CONFIG,
    scene=SimulationScene(os.environ.get(SCENE_ENVIRONMENT_VARIABLE, SimulationScene.EMPTY)),
)
