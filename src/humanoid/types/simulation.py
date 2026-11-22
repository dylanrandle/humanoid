"""Configuration types for the native MuJoCo digital twin."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from humanoid.types.robot import RobotName

if TYPE_CHECKING:
    import mujoco


type RenderFrame = Callable[[mujoco.MjData, float], None]


class SimulationScene(StrEnum):
    EMPTY = "empty"
    TOY_CLEANUP = "toy_cleanup"


@dataclass(frozen=True, kw_only=True)
class MujocoSimulationConfig:
    """Physics and actuator defaults shared by every generated robot model."""

    scene: SimulationScene = SimulationScene.EMPTY
    physics_timestep: float = 0.005
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
    enable_cameras: bool = True

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


@dataclass(frozen=True)
class SimulationState:
    """A native MuJoCo integration state and the model needed to interpret it."""

    robot: RobotName
    scene: SimulationScene
    state: np.ndarray

    def __post_init__(self) -> None:
        if self.state.ndim != 1 or not self.state.size or not np.isfinite(self.state).all():
            raise ValueError("Expected a finite, nonempty simulation state vector.")
