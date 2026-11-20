"""Inputs, configuration and status for camera-guided pickup."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.types.camera import CameraFrames, CameraPreview


@dataclass(frozen=True, kw_only=True)
class PickConfig:
    segmenter_model: str
    segmenter_revision: str
    segmentation_image_size: int
    detection_threshold: float
    tool_tip_offset: tuple[float, float, float]
    gripper_height: float
    max_grasp_candidates: int
    open_gap: float
    gap_per_radian: float
    squeeze: float
    approach_clearance: float
    lift_height: float
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    linear_speed: float
    angular_speed: float
    gripper_speed: float
    rate_hz: float
    phase_timeout: float
    feedback_timeout: float
    image_max_age: float
    position_tolerance: float
    orientation_tolerance: float
    squeeze_seconds: float
    minimum_points: int


@dataclass(frozen=True, kw_only=True)
class ObjectMask:
    mask: NDArray[np.bool_]
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True, kw_only=True)
class CameraPointCloud:
    """Native optical XYZ; (-1, -1) denotes a point invisible in the RGB view."""

    points: NDArray[np.float32]
    rgb_pixels: NDArray[np.int64]
    base_from_camera: pin.SE3

    @property
    def base_points(self) -> NDArray[np.float64]:
        return self.points @ self.base_from_camera.rotation.T + self.base_from_camera.translation


@dataclass(frozen=True, kw_only=True)
class GraspProposal:
    """Depth optical pose; +X approach, +Y opening, tip at pose + depth * X.

    Width is the required jaw gap, predicted or geometrically estimated. A missing
    width permits pose preview only; the executor rejects unresolved closure.
    """

    pose: pin.SE3
    width: float | None
    height: float
    depth: float
    score: float


@dataclass(frozen=True, kw_only=True)
class GraspGenXConfig:
    finger_meshes: tuple[str, str]
    gripper_links: tuple[str, ...]
    num_grasps: int
    minimum_score: float
    collision_resolution: float
    approach_step: float


@dataclass(frozen=True, kw_only=True)
class PickWaypoint:
    name: str
    pose: pin.SE3
    gripper: float
    hold_seconds: float = 0.0


@dataclass(frozen=True, kw_only=True)
class PickPlan:
    grasp: GraspProposal
    object_points: NDArray[np.float64]
    waypoints: tuple[PickWaypoint, ...]


@dataclass(frozen=True, kw_only=True)
class PickPerceptionPreview:
    frames: CameraFrames
    cloud: CameraPointCloud
    rgb_from_depth: pin.SE3
    object_description: str
    masks: tuple[ObjectMask, ...]
    grasps: tuple[GraspProposal, ...] = ()
    selected: GraspProposal | None = None


@dataclass(frozen=True, kw_only=True)
class PickPreview:
    """Encoded once on the perception thread; cheap to serve from the control socket."""

    timestamp: float
    rgb: CameraPreview
    depths: dict[float, CameraPreview]
    object_description: str
    mask_count: int
    grasp_count: int
    selected: bool


class GraspInference(Protocol):
    def grasp(
        self, points: np.ndarray, conditioning: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


class PickPhase(StrEnum):
    READY = "ready"
    PLANNING = "planning"
    APPROACH = "approach"
    GRASP = "grasp"
    SQUEEZE = "squeeze"
    LIFT = "lift"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, kw_only=True)
class PickStatus:
    phase: PickPhase
    message: str
    object_description: str = ""
    busy: bool = False
    models_ready: bool = False
