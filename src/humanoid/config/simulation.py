"""Native MuJoCo simulation configuration defaults."""

import os

from humanoid.constants import SCENE_ENVIRONMENT_VARIABLE
from humanoid.types.simulation import MujocoSimulationConfig, SimulationScene

DEFAULT_MUJOCO_SIMULATION_CONFIG = MujocoSimulationConfig(
    scene=SimulationScene(os.environ.get(SCENE_ENVIRONMENT_VARIABLE, SimulationScene.EMPTY)),
)
