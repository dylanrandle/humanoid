"""Configuration types for the native MuJoCo digital twin."""

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from humanoid.types.robot import RobotJointCommand, RobotState


class MujocoIntegrator(StrEnum):
    EULER = "Euler"
    RK4 = "RK4"
    IMPLICIT = "implicit"
    IMPLICIT_FAST = "implicitfast"


@dataclass(frozen=True, kw_only=True)
class MujocoSimulationConfig:
    """Physics and actuator defaults shared by every generated robot model."""

    physics_timestep: float = 0.005
    integrator: MujocoIntegrator = MujocoIntegrator.IMPLICIT_FAST
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
class SimulationReset:
    timestamp: float
    request_id: str
    task: str
    seed: int


@dataclass(frozen=True)
class SimulationResetStatus:
    timestamp: float
    request_id: str
    error: str = ""


@dataclass(frozen=True)
class SimulationState:
    """One simulator snapshot; its command is the last applied actuator target.

    Sensor/action timestamps use host_monotonic. sim_seconds is the independent
    physics clock. qpos/qvel let privileged experts render and inspect a local
    copy of the world without advancing or controlling its physics.
    """

    timestamp: float
    reset_id: str
    sim_seconds: float
    qpos: np.ndarray
    qvel: np.ndarray
    state: RobotState
    command: RobotJointCommand
    error: str = ""
