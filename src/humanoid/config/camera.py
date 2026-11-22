"""Base and wrist RGB-D camera configurations."""

from humanoid.types.camera import CameraPreviewConfig, RealSenseCameraConfig, SimulatedCameraConfig
from humanoid.types.depth import DepthCamera

REALSENSE_CAMERA_CONFIG = RealSenseCameraConfig(
    width=640,
    height=480,
    fps=30,
    rgb_frame="ego_camera_color_optical_frame",
    depth_frame="ego_camera_depth_optical_frame",
    frame_timeout_seconds=3.0,
)

# Nominal pinhole views; hardware publishes its device-specific calibration.
SIMULATED_CAMERA_CONFIG = SimulatedCameraConfig(
    model=DepthCamera.D435,
    width=640,
    height=480,
    fps=30,
    rgb_frame="ego_camera_color_optical_frame",
    depth_frame="ego_camera_depth_optical_frame",
    rgb_vertical_fov=42.0,
    depth_vertical_fov=58.0,
    depth_scale=0.001,
)

CAMERA_PREVIEW_CONFIG = CameraPreviewConfig(
    max_age_seconds=2.0,
    max_width=640,
    jpeg_quality=80,
    depth_near_meters=0.2,
    depth_ranges_meters=(1.0, 3.0, 5.0, 10.0),
    default_depth_far_meters=3.0,
)

# Nominal VGA pinhole view for the wrist-mounted close-range module.
WRIST_D405_CONFIG = SimulatedCameraConfig(
    model=DepthCamera.D405,
    width=640,
    height=480,
    fps=30,
    rgb_frame="wrist_camera_color_optical_frame",
    depth_frame="wrist_camera_depth_optical_frame",
    rgb_vertical_fov=58,
    depth_vertical_fov=58,
    depth_scale=0.0001,
)
