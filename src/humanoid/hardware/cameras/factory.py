"""Construct a physical camera source from its configured hardware type."""

from humanoid.hardware.cameras.realsense import RealSenseCamera
from humanoid.types.camera import CameraSource, HardwareCameraConfig, RealSenseCameraConfig


def create_camera(config: HardwareCameraConfig) -> CameraSource:
    if isinstance(config, RealSenseCameraConfig):
        return RealSenseCamera(config)
    raise TypeError(f"Unsupported camera hardware config: {type(config).__name__}")
