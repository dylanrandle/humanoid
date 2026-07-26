"""Configuration types for the native MuJoCo digital twin."""

import math
import os
from dataclasses import dataclass
from enum import StrEnum


class MujocoScene(StrEnum):
    """Named MuJoCo environments selectable by the operator."""

    EMPTY = "empty"
    FLOOR_AND_CUBE = "floor-and-cube"

    @classmethod
    def from_environment(cls) -> "MujocoScene":
        """Parse the selected scene, using the configured default when unset."""
        from humanoid.constants import (  # noqa: PLC0415
            DEFAULT_MUJOCO_SCENE,
            MUJOCO_SCENE_ENVIRONMENT_VARIABLE,
        )

        value = os.getenv(MUJOCO_SCENE_ENVIRONMENT_VARIABLE)
        if value is None or not value.strip():
            return cls(DEFAULT_MUJOCO_SCENE)
        return cls(value.lower().strip())


@dataclass(frozen=True, kw_only=True)
class MujocoSimulationConfig:
    """Physics and actuator defaults shared by every generated robot model."""

    physics_timestep: float = 0.001
    publish_rate_hz: float = 500.0
    position_kp: float = 1000.0
    position_damping_ratio: float = 1.0
    velocity_kv: float = 2.0
    joint_armature: float = 0.01
    joint_damping: float = 0.1
    root_velocity_kv: float = 50.0
    root_force_limit: float = 100.0
    minimum_body_mass: float = 0.001
    minimum_body_inertia: float = 1e-6

    def __post_init__(self) -> None:
        positive_values = (
            self.physics_timestep,
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


@dataclass(frozen=True, kw_only=True)
class FloorAndCubeSceneConfig:
    """Dimensions and placement for the ``floor-and-cube`` scene."""

    floor_half_extent: float = 2.0
    cube_edge_length: float = 0.04
    cube_mass: float = 0.05
    cube_x: float = 0.3
    cube_y: float = 0.0

    def __post_init__(self) -> None:
        positive_values = (
            self.floor_half_extent,
            self.cube_edge_length,
            self.cube_mass,
        )
        for value in positive_values:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"MuJoCo scene {value=} must be positive and finite.")
        if not math.isfinite(self.cube_x) or not math.isfinite(self.cube_y):
            raise ValueError("MuJoCo cube position must be finite.")
