"""Camera configuration, native images, and the acquisition interface."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

MAX_VERTICAL_FOV_DEGREES = 180


class CameraEncoding(StrEnum):
    RGB8 = "rgb8"
    DEPTH16 = "16UC1"


@dataclass(frozen=True, kw_only=True)
class CameraConfig:
    width: int
    height: int
    fps: int
    rgb_frame: str
    depth_frame: str

    def __post_init__(self) -> None:
        for name in ("width", "height", "fps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Camera {name} must be a positive integer.")
        if not self.rgb_frame.strip() or not self.depth_frame.strip():
            raise ValueError("Camera optical frame names must not be empty.")
        if self.rgb_frame == self.depth_frame:
            raise ValueError("Native RGB and depth require distinct optical frames.")


@dataclass(frozen=True, kw_only=True)
class RealSenseCameraConfig(CameraConfig):
    frame_timeout_seconds: float
    serial_number: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.frame_timeout_seconds) or self.frame_timeout_seconds <= 0:
            raise ValueError("Camera frame timeout must be positive and finite.")
        if self.serial_number is not None and not self.serial_number.strip():
            raise ValueError("Camera serial number must not be empty.")


@dataclass(frozen=True, kw_only=True)
class SimulatedCameraConfig(CameraConfig):
    rgb_vertical_fov: float
    depth_vertical_fov: float
    depth_scale: float

    def __post_init__(self) -> None:
        super().__post_init__()
        for fov in (self.rgb_vertical_fov, self.depth_vertical_fov):
            if not math.isfinite(fov) or not 0 < fov < MAX_VERTICAL_FOV_DEGREES:
                raise ValueError("Camera vertical field of view must be between 0 and 180 degrees.")
        if not math.isfinite(self.depth_scale) or self.depth_scale <= 0:
            raise ValueError("Camera depth scale must be positive and finite.")


@dataclass(frozen=True, kw_only=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera dimensions must be positive.")
        if not all(math.isfinite(x) for x in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("Camera intrinsics must be finite.")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera focal lengths must be positive.")
        if not all(math.isfinite(x) for x in self.distortion_coefficients):
            raise ValueError("Camera distortion coefficients must be finite.")


@dataclass(frozen=True, kw_only=True)
class CameraImage:
    """Native pixels and calibration; depth in meters is data * depth_scale.

    timestamp is host monotonic receipt time in seconds (shared by a frameset).
    device_timestamp is capture time in seconds in timestamp_domain, not assumed
    comparable with the host clock. Zero depth samples mean invalid depth.
    """

    timestamp: float
    device_timestamp: float
    timestamp_domain: str
    frame_number: int
    frame_id: str
    serial_number: str
    encoding: CameraEncoding
    data: np.ndarray
    intrinsics: CameraIntrinsics
    depth_scale: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.encoding, CameraEncoding):
            raise ValueError("Camera encoding must be a CameraEncoding.")
        shape = (self.intrinsics.height, self.intrinsics.width)
        if self.encoding is CameraEncoding.RGB8:
            shape = (*shape, 3)
            dtype = np.dtype("uint8")
            if self.depth_scale != 0:
                raise ValueError("RGB images must have zero depth scale.")
        else:
            dtype = np.dtype("uint16")
            if not math.isfinite(self.depth_scale) or self.depth_scale <= 0:
                raise ValueError("Depth images require a positive finite depth scale.")
        if self.data.shape != shape or self.data.dtype != dtype:
            raise ValueError(f"Expected camera pixels with shape {shape} and dtype {dtype}.")
        if not self.frame_id.strip():
            raise ValueError("Camera optical frame must not be empty.")
        if not math.isfinite(self.timestamp) or not math.isfinite(self.device_timestamp):
            raise ValueError("Camera timestamps must be finite.")


@dataclass(frozen=True, kw_only=True)
class CameraFrames:
    rgb: CameraImage
    depth: CameraImage


class CameraSource(Protocol):
    def start(self) -> None: ...

    def read(self) -> CameraFrames | None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, kw_only=True)
class CameraPreviewConfig:
    max_age_seconds: float
    max_width: int
    jpeg_quality: int
    depth_near_meters: float
    depth_ranges_meters: tuple[float, ...]
    default_depth_far_meters: float


@dataclass(frozen=True, kw_only=True)
class CameraPreview:
    connected: bool
    age_seconds: float | None = None
    image: str | None = None
    width: int | None = None
    height: int | None = None
    frame_id: str | None = None
    frame_number: int | None = None


@dataclass(frozen=True, kw_only=True)
class CameraInferencePreview:
    """Annotated source frames, never annotations over unrelated live images."""

    rgb: CameraPreview
    depth: CameraPreview
    object_description: str
    mask_count: int
    grasp_count: int
    selected: bool


@dataclass(frozen=True, kw_only=True)
class CameraPreviewSnapshot:
    rgb: CameraPreview
    depth: CameraPreview
    depth_near_meters: float
    depth_far_meters: float
    inference: CameraInferencePreview | None = None
