"""Camera configuration, native images, and the acquisition interface."""

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Literal, NewType, Protocol

import numpy as np

from humanoid.types.depth import DepthCamera

MAX_VERTICAL_FOV_DEGREES = 180


class DepthView(StrEnum):
    ALIGNED = "aligned"
    NATIVE = "native"


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
class HardwareCameraConfig(CameraConfig):
    serial_number: str | None = None
    expected_model: DepthCamera | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.serial_number is not None and not self.serial_number.strip():
            raise ValueError("Camera serial number must not be empty.")


@dataclass(frozen=True, kw_only=True)
class RealSenseCameraConfig(HardwareCameraConfig):
    frame_timeout_seconds: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.frame_timeout_seconds) or self.frame_timeout_seconds <= 0:
            raise ValueError("Camera frame timeout must be positive and finite.")


@dataclass(frozen=True, kw_only=True)
class SimulatedCameraConfig(CameraConfig):
    model: DepthCamera
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


ObservationId = NewType("ObservationId", str)


@dataclass(frozen=True, kw_only=True)
class CaptureMetadata:
    observation_id: ObservationId
    camera_model: DepthCamera | None
    serial_number: str
    rgb_frame_id: str
    depth_frame_id: str | None
    rgb_frame_number: int
    depth_frame_number: int | None
    depth_scale_m_per_unit: float | None
    rgb_device_timestamp: float
    depth_device_timestamp: float | None
    rgb_timestamp_domain: str
    depth_timestamp_domain: str | None
    intrinsics: CameraIntrinsics
    depth_units: Literal["metres"] = "metres"


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
class RegisteredDepth:
    camera_model: DepthCamera
    rgb_frame_number: int
    image: CameraImage

    def matches(self, rgb: CameraImage, depth: CameraImage) -> bool:
        return (
            self.image.encoding is CameraEncoding.DEPTH16
            and rgb.encoding is CameraEncoding.RGB8
            and depth.encoding is CameraEncoding.DEPTH16
            and self.image.timestamp == rgb.timestamp == depth.timestamp
            and self.image.serial_number == rgb.serial_number == depth.serial_number
            and self.image.frame_id == rgb.frame_id
            and self.rgb_frame_number == rgb.frame_number
            and self.image.frame_number == depth.frame_number
            and self.image.device_timestamp == depth.device_timestamp
            and self.image.timestamp_domain == depth.timestamp_domain
            and self.image.intrinsics == rgb.intrinsics
            and self.image.depth_scale == depth.depth_scale
        )


@dataclass(frozen=True, kw_only=True)
class CameraView:
    """Selected camera view. Missing aligned measurements leave RGB available."""

    rgb: CameraImage
    depth: CameraImage | None
    registered: RegisteredDepth | None = None

    def __post_init__(self) -> None:
        if self.registered is not None and (
            self.depth is None or not self.registered.matches(self.rgb, self.depth)
        ):
            raise ValueError("Registered depth does not belong to this RGB-D capture.")


@dataclass(frozen=True, kw_only=True)
class CameraFrames(CameraView):
    """Driver frameset containing native depth and optional depth registered to RGB."""

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
class CameraPreviewSnapshot:
    rgb: CameraPreview
    depth: CameraPreview
    depth_near_meters: float
    depth_far_meters: float


@dataclass(frozen=True, kw_only=True)
class RobotCameraConfig:
    simulation: SimulatedCameraConfig | None = None
    hardware: HardwareCameraConfig | None = None


@dataclass(frozen=True)
class CameraTopic[T]:
    value: str
    message_type: type[T]
    encoding: CameraEncoding | None = None


@dataclass(frozen=True)
class CameraTopics:
    rgb: CameraTopic[CameraImage]
    depth: CameraTopic[CameraImage]
    registered_depth: CameraTopic[RegisteredDepth]


@cache
def camera_topics(name: str) -> CameraTopics:
    """Map a configured camera name to its uppercase LCM channel family."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Invalid camera name {name!r}. Use a name matching [A-Za-z][A-Za-z0-9_]*."
        )
    prefix = f"CAMERA/{name.upper()}"
    return CameraTopics(
        CameraTopic(f"{prefix}/RGB", CameraImage, CameraEncoding.RGB8),
        CameraTopic(f"{prefix}/DEPTH", CameraImage, CameraEncoding.DEPTH16),
        CameraTopic(f"{prefix}/REGISTERED_DEPTH", RegisteredDepth),
    )
