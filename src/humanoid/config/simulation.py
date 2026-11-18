"""Native MuJoCo simulation configuration defaults."""

from humanoid.config.camera import SIMULATED_CAMERA_CONFIG
from humanoid.types.simulation import MujocoSimulationConfig

# Stiffen tangential contact constraints to reduce numerical slip during holds.
# https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip
DEFAULT_MUJOCO_SIMULATION_CONFIG = MujocoSimulationConfig(
    camera=SIMULATED_CAMERA_CONFIG,
    contact_impedance_ratio=10.0,
    # Grip tape on the rigid FinRay surfaces.
    mesh_sliding_friction={"finger_1_fin": 2.0, "finger_2_fin": 2.0},
)
