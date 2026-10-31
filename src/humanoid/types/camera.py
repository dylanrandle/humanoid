"""Calibration shapes for simulated RGB sensors."""

from dataclasses import dataclass

import numpy as np

MAX_CAMERA_FOV_DEGREES = 180


@dataclass(frozen=True)
class CameraMountConfig:
    """Bracket plate and screw centers in the camera's parent frame.

    Plate axes are right/outward/up; screws point along its outward axis.
    """

    position: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    size: tuple[float, float, float]
    holes: tuple[tuple[float, float, float], ...]
    screw_radius: float = 0.0015
    screw_length: float = 0.004
    mass: float = 0.008


@dataclass(frozen=True)
class RGBCameraConfig:
    """Calibrated optical frame (right/down/forward), relative to its parent.

    A missing parent means world coordinates. The mount transform uses the
    optical right/down/forward convention.
    """

    position: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    width: int = 640
    height: int = 480
    vertical_fov_degrees: float = 55.0
    parent_frame: str | None = None
    horizontal_fov_degrees: float | None = None
    housing_size: tuple[float, float, float] | None = None
    housing_mass: float = 0.058
    mount: CameraMountConfig | None = None

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3)):
            raise ValueError("Camera rotation must be an orthonormal 3x3 matrix.")
        if not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError("Camera rotation must be right-handed.")
        if np.asarray(self.position).shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError("Camera position must contain three finite coordinates.")
        if (
            self.width <= 0
            or self.height <= 0
            or not 0 < self.vertical_fov_degrees < MAX_CAMERA_FOV_DEGREES
        ):
            raise ValueError("Camera dimensions and field of view must be positive and valid.")
        if self.horizontal_fov_degrees is not None and not (
            0 < self.horizontal_fov_degrees < MAX_CAMERA_FOV_DEGREES
        ):
            raise ValueError("Horizontal field of view must be valid.")
        if self.housing_size is not None and (
            np.asarray(self.housing_size).shape != (3,)
            or not np.isfinite(self.housing_size).all()
            or min(self.housing_size) <= 0
            or not np.isfinite(self.housing_mass)
            or self.housing_mass <= 0
        ):
            raise ValueError("Camera housing dimensions and mass must be positive and finite.")

    @property
    def focal_length(self) -> float:
        return self.height / (2 * np.tan(np.deg2rad(self.vertical_fov_degrees) / 2))

    @property
    def focal_x(self) -> float:
        if self.horizontal_fov_degrees is None:
            return self.focal_length
        return self.width / (2 * np.tan(np.deg2rad(self.horizontal_fov_degrees) / 2))


@dataclass(frozen=True)
class CameraFrame:
    """A rectified RGB JPEG and its calibration, without privileged extrinsics.

    timestamp is capture time in seconds on the collection clock, not arrival
    time. Hardware producers must map their device clock to that clock and
    rectify images before publishing. Intrinsics describe the stored image;
    frame_id names its optical frame (x right, y down, z forward).
    """

    timestamp: float
    name: str
    frame_id: str
    width: int
    height: int
    intrinsics: np.ndarray
    rgb_jpeg: bytes
