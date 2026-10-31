"""Shared scene geometry payloads for visualization and replay."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pinocchio as pin

type SceneGeometryKind = Literal["box", "cylinder", "plane"]


@dataclass(frozen=True)
class SceneGeometry:
    """Privileged display geometry. Never an autonomous-policy observation."""

    name: str
    kind: SceneGeometryKind
    size: np.ndarray
    rgba: np.ndarray
    pose: pin.SE3


@dataclass(frozen=True)
class SimulationSceneState:
    """Visualization-only snapshot, separate from robot and camera sensors."""

    timestamp: float
    geoms: tuple[SceneGeometry, ...]
