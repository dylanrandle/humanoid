"""Configuration types for the native MuJoCo digital twin."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from humanoid.types.camera import SimulatedCameraConfig


@dataclass(frozen=True, kw_only=True)
class MujocoSimulationConfig:
    """Physics and actuator defaults shared by every generated robot model."""

    physics_timestep: float = 0.005
    contact_impedance_ratio: float = 1.0
    mesh_sliding_friction: Mapping[str, float] = field(default_factory=dict)
    publish_rate_hz: float = 200.0
    position_kp: float = 1000.0
    position_damping_ratio: float = 1.0
    velocity_kv: float = 2.0
    joint_armature: float = 0.01
    joint_damping: float = 0.1
    root_velocity_kv: float = 50.0
    root_force_limit: float = 100.0
    minimum_body_mass: float = 0.001
    minimum_body_inertia: float = 1e-6
    camera: SimulatedCameraConfig | None = None

    def __post_init__(self) -> None:
        positive_values = (
            self.physics_timestep,
            self.contact_impedance_ratio,
            self.publish_rate_hz,
            self.position_kp,
            self.position_damping_ratio,
            self.velocity_kv,
            self.joint_armature,
            self.joint_damping,
            self.root_velocity_kv,
            self.root_force_limit,
            self.minimum_body_mass,
            self.minimum_body_inertia,
        )
        for value in positive_values:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"MuJoCo {value=} must be positive and finite.")
        for mesh_name, friction in self.mesh_sliding_friction.items():
            if not mesh_name.strip() or not math.isfinite(friction) or friction < 0.0:
                raise ValueError(
                    "Mesh friction requires a nonempty name and a finite nonnegative value."
                )
