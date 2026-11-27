"""Native MuJoCo simulation configuration defaults."""

import os

from humanoid.constants import EnvironmentVariable
from humanoid.types.simulation import MujocoSimulationConfig, SimulationScene

DEFAULT_MUJOCO_SIMULATION_CONFIG = MujocoSimulationConfig(
    scene=SimulationScene(os.environ.get(EnvironmentVariable.SCENE, SimulationScene.EMPTY)),
)
