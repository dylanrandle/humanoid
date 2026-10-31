"""Scene, expert configuration, and planning shapes for the cleanup task."""

from dataclasses import dataclass

import numpy as np

from humanoid.types.camera import RGBCameraConfig


@dataclass(frozen=True)
class CleanupConfig:
    grasp_offset: tuple[float, float, float]
    open_position: float
    closed_position: float
    open_width: float
    width_per_radian: float
    grasp_seed_arm_positions: tuple[float, ...]
    drop_slot_offsets: tuple[tuple[float, float], ...]
    transit_height: float = 0.10
    release_height: float = 0.07
    max_attempts: int = 8
    motion_timeout: float = 14.0
    cartesian_speed: float = 0.06
    joint_speed: float = 0.8
    position_tolerance: float = 0.006
    orientation_tolerance: float = 0.08
    min_holding_width: float = 0.004


@dataclass(frozen=True)
class MobileCleanupConfig:
    navigation_min: tuple[float, float]
    navigation_max: tuple[float, float]
    pickup_offset: tuple[float, float]
    footprint_radius: float = 0.20
    clearance: float = 0.015
    obstacle_sides: int = 32
    max_speed: float = 0.08
    max_acceleration: float = 0.12
    max_yaw_speed: float = 0.6
    max_yaw_acceleration: float = 0.6
    position_gain: float = 1.5
    yaw_gain: float = 3.0
    position_tolerance: float = 0.003
    yaw_tolerance: float = 0.025
    navigation_timeout: float = 60.0
    waypoint_tolerance: float = 0.003


@dataclass(frozen=True)
class SceneObjectConfig:
    name: str
    kind: str
    position: tuple[float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float, float]
    yaw: float = 0.0
    mass: float = 0.025


@dataclass(frozen=True)
class CleanupSceneConfig:
    floor_height: float
    bin_position: tuple[float, float]
    bin_half_size: tuple[float, float]
    bin_height: float
    wall_thickness: float
    objects: tuple[SceneObjectConfig, ...]
    cameras: dict[str, RGBCameraConfig]
    friction_impedance_ratio: float = 100.0
    noslip_iterations: int = 20


@dataclass(frozen=True)
class CleanupViewConfig:
    """Thresholds for optional RGB visibility diagnostics."""

    frame_margin: float = 12.0
    min_box_pixels: float = 64.0
    min_visible_fraction: float = 0.20
    min_bin_visible_fraction: float = 0.15
    min_held_visible_fraction: float = 0.10
    min_held_pixels: float = 256.0
    ray_grid_size: int = 9


@dataclass(frozen=True)
class TargetVisibility:
    """Privileged image-space QA; never part of the policy observation."""

    target: str
    in_frame: bool
    box_pixels: float
    visible_fraction: float
    center: tuple[float, float]
    visible: bool
    camera: str
    covering_cameras: tuple[str, ...] = ()

    @property
    def covered(self) -> bool:
        """The entire box is framed by one view or the named overlapping views."""
        return self.in_frame or bool(self.covering_cameras)


@dataclass(frozen=True)
class NavigationObstacle:
    """Counterclockwise convex polygon and outward halfspaces Ax + b <= 0."""

    vertices: np.ndarray
    halfspaces: np.ndarray


@dataclass(frozen=True)
class NavigationMap:
    lower: np.ndarray
    upper: np.ndarray
    obstacles: tuple[NavigationObstacle, ...]


@dataclass(frozen=True)
class NavigationPlan:
    """Shortest translation route to a facing approach pose."""

    waypoints: tuple[np.ndarray, ...]
    target: np.ndarray
    yaw: float
    length: float
