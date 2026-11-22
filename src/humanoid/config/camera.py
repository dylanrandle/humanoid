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

# D400 Series Datasheet (337029-017), camera SKU properties: D435 VGA depth.
# The RGB sensor's 42.5° vertical view gives 54.8° horizontally for a 4:3 crop.
SIMULATED_CAMERA_CONFIG = SimulatedCameraConfig(
    model=DepthCamera.D435,
    width=640,
    height=480,
    fps=30,
    rgb_frame="ego_camera_color_optical_frame",
    depth_frame="ego_camera_depth_optical_frame",
    rgb_fov_degrees=(54.8, 42.5),
    depth_fov_degrees=(75.0, 62.0),
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

# The datasheet specifies D405 depth at 16:9 and lists VGA as N/A.
# Use this wrist unit's recorded VGA stream intrinsics for the 4:3 views.
WRIST_D405_CONFIG = SimulatedCameraConfig(
    model=DepthCamera.D405,
    width=640,
    height=480,
    fps=30,
    rgb_frame="wrist_camera_color_optical_frame",
    depth_frame="wrist_camera_depth_optical_frame",
    rgb_fov_degrees=(79.15, 63.63),
    depth_fov_degrees=(80.11, 64.47),
    depth_scale=0.0001,
)
