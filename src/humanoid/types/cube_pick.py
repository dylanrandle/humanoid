"""Configuration and perception results for the single-cube demonstration."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class CubePickPhase(StrEnum):
    SEARCH = "search"
    APPROACH = "approach"
    DESCEND = "descend"
    SQUEEZE = "squeeze"
    DONE = "done"
    FAILED = "failed"


class CubeDetectionMethod(StrEnum):
    RED_TOP = "red_top"
    DEPTH_SIDES = "depth_sides"


@dataclass(frozen=True, kw_only=True)
class CubePickConfig:
    cube_size: float
    ground_height: float
    grip_offset: tuple[float, float, float]
    approach_clearance: float
    position_tolerance: float
    orientation_tolerance: float
    open_position: float
    closed_position: float
    squeeze_seconds: float
    phase_timeout: float
    image_max_age: float
    minimum_points: int
    top_tolerance: float
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    detection_method: CubeDetectionMethod = CubeDetectionMethod.RED_TOP

    def __post_init__(self) -> None:
        for name in (
            "cube_size",
            "approach_clearance",
            "position_tolerance",
            "orientation_tolerance",
            "squeeze_seconds",
            "phase_timeout",
            "image_max_age",
            "top_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.minimum_points < 1:
            raise ValueError("minimum_points must be positive.")
        for value in (
            self.ground_height,
            self.open_position,
            self.closed_position,
            *self.grip_offset,
            *self.workspace_min,
            *self.workspace_max,
        ):
            if not np.isfinite(value):
                raise ValueError("Cube picking configuration must be finite.")
        if np.any(np.asarray(self.workspace_min) >= self.workspace_max):
            raise ValueError("Workspace minimum must be below maximum.")


@dataclass(frozen=True, kw_only=True)
class CubeDetection:
    """Cube estimate in the robot's tool-command (base) frame."""

    center: np.ndarray
    yaw: float
    point_count: int
