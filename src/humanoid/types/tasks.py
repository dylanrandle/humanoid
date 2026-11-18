"""Task contracts shared by trusted simulation runners and evaluators."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    import mujoco


@dataclass(frozen=True, kw_only=True)
class TaskEvaluation:
    """Private evaluation result; only task_complete is exposed to an exploring agent."""

    task_complete: bool
    objects_delivered: int
    objects_total: int
    reason: str


@dataclass(frozen=True, kw_only=True)
class ToyCarCleanupConfig:
    room_min: tuple[float, float]
    room_max: tuple[float, float]
    ground_z: float
    car_count: int
    car_length: float
    car_width: float
    car_height: float
    car_mass: float
    bin_center: tuple[float, float]
    bin_inner_size: tuple[float, float]
    bin_rim_height: float
    bin_wall_thickness: float
    settle_seconds: float

    def __post_init__(self) -> None:
        if type(self.car_count) is not int or self.car_count < 1:
            raise ValueError("car_count must be a positive integer.")
        for name in (
            "car_length",
            "car_width",
            "car_height",
            "car_mass",
            "bin_rim_height",
            "bin_wall_thickness",
            "settle_seconds",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for values in (self.room_min, self.room_max, self.bin_center, self.bin_inner_size):
            if np.shape(values) != (2,) or not np.isfinite(values).all():
                raise ValueError("Scene coordinates must be finite 2D vectors.")
        if not np.isfinite(self.ground_z) or np.any(np.asarray(self.room_min) >= self.room_max):
            raise ValueError("Invalid floor or room bounds.")
        if np.any(np.asarray(self.bin_inner_size) <= max(self.car_length, self.car_width)):
            raise ValueError("The bin must fit a car.")
        if self.bin_rim_height <= self.car_height + self.bin_wall_thickness:
            raise ValueError("The bin must contain a car below its rim.")


class SimulationTask(Protocol):
    """Host-owned task logic, never installed in an agent's writable workspace."""

    name: str
    instruction: str

    def build_scene(self, spec: "mujoco.MjSpec") -> None: ...
    def reset(self, model: "mujoco.MjModel", data: "mujoco.MjData", seed: int) -> None: ...
    def evaluate(self, model: "mujoco.MjModel", data: "mujoco.MjData") -> TaskEvaluation: ...


@dataclass(frozen=True, kw_only=True)
class SimulationTaskSelection:
    name: str
    seed: int
    formal: bool = False

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0 or type(self.formal) is not bool:
            raise ValueError("Task seed must be nonnegative and formal must be boolean.")
