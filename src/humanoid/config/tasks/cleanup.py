"""Cleanup scene and Triskel simulation-expert defaults."""

from dataclasses import replace

import numpy as np

from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.types.camera import CameraMountConfig, RGBCameraConfig
from humanoid.types.simulation import MujocoIntegrator
from humanoid.types.tasks.cleanup import (
    CleanupConfig,
    CleanupSceneConfig,
    CleanupViewConfig,
    MobileCleanupConfig,
    SceneObjectConfig,
)

CLEANUP_SIMULATION_CONFIG = replace(
    DEFAULT_MUJOCO_SIMULATION_CONFIG,
    # Euler avoids the dynamic grasp slip from implicit actuator damping. A
    # 0.25 ms step also resolves the fast base yaw servo without overshoot.
    physics_timestep=0.00025,
    integrator=MujocoIntegrator.EULER,
)

WRIST_CAMERA_NAME = "wrist"
BASE_CAMERA_YAWS = {
    "base_1_2": -np.pi / 6,
    "base_2_3": np.pi / 2,
    "base_3_1": 7 * np.pi / 6,
}

# RGB camera at 848x480 with a nominal 87x58 degree FOV. Hole centers measured
# from arm_6.stl, after its URDF visual transform: two 3.25 mm bores, 20.5 mm
# apart, behind the final gripper roll joint.
# The simplified bracket raises the camera above these tabs. Its lens extrinsics
# remain configurable; they are not a measured physical hand-eye calibration.
_MOUNT_TILT = np.deg2rad(25.0)
_MOUNT_HOLES = np.array(
    [
        [0.0112804, 0.0590720, 0.0288063],
        [0.0317742, 0.0587174, 0.0284471],
    ]
)
_MOUNT_RIGHT = _MOUNT_HOLES[1] - _MOUNT_HOLES[0]
_MOUNT_RIGHT /= np.linalg.norm(_MOUNT_RIGHT)
_MOUNT_UP = np.cross(_MOUNT_RIGHT, [0.01730, 0.99985, -0.000303])
_MOUNT_UP /= np.linalg.norm(_MOUNT_UP)
_MOUNT_FORWARD = np.cross(_MOUNT_UP, _MOUNT_RIGHT)
_MOUNT_ROTATION = np.column_stack([_MOUNT_RIGHT, _MOUNT_FORWARD, _MOUNT_UP])
_OPTICAL_ROTATION = _MOUNT_ROTATION @ np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -np.sin(_MOUNT_TILT), np.cos(_MOUNT_TILT)],
        [0.0, -np.cos(_MOUNT_TILT), -np.sin(_MOUNT_TILT)],
    ]
)
TRISKEL_WRIST_CAMERA = RGBCameraConfig(
    parent_frame="arm_6_link",
    position=tuple(_MOUNT_HOLES.mean(axis=0) + 0.010 * _MOUNT_FORWARD + 0.026 * _MOUNT_UP),
    rotation=tuple(tuple(row) for row in _OPTICAL_ROTATION),
    mount=CameraMountConfig(
        position=tuple(_MOUNT_HOLES.mean(axis=0) + 0.002 * _MOUNT_FORWARD + 0.010 * _MOUNT_UP),
        rotation=tuple(tuple(row) for row in _MOUNT_ROTATION),
        size=(0.026, 0.003, 0.028),
        holes=tuple(tuple(row) for row in _MOUNT_HOLES),
    ),
    width=848,
    height=480,
    horizontal_fov_degrees=87.0,
    vertical_fov_degrees=58.0,
    housing_size=(0.042, 0.042, 0.023),
)


def _base_camera(yaw: float) -> RGBCameraConfig:
    """Mount between the wheels in the gap above the bottom plate."""
    outward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    rotation = np.column_stack([right, np.cross(outward, right), outward])
    return replace(
        TRISKEL_WRIST_CAMERA,
        parent_frame="lower_plate_link",
        # The lower plate ends at z=7 mm; the upper plate starts at z=53 mm.
        # This puts the complete housing between them.
        position=tuple(0.108 * outward + np.array([0.0, 0.0, 0.029])),
        rotation=tuple(tuple(row) for row in rotation),
        # Overlapping wide views cover the surroundings from three headings.
        width=1920,
        height=1080,
        horizontal_fov_degrees=150.0,
        vertical_fov_degrees=90.0,
        mount=CameraMountConfig(
            position=tuple(0.094 * outward + np.array([0.0, 0.0, 0.0095])),
            rotation=tuple(tuple(row) for row in np.column_stack([right, outward, [0, 0, 1]])),
            size=(0.030, 0.006, 0.005),
            holes=(),
        ),
    )


TRISKEL_CLEANUP_CAMERAS = {
    WRIST_CAMERA_NAME: TRISKEL_WRIST_CAMERA,
    **{name: _base_camera(yaw) for name, yaw in BASE_CAMERA_YAWS.items()},
}

TRISKEL_CLEANUP_CONFIG = CleanupConfig(
    grasp_offset=(0.0015, 0.090, 0.0),
    open_position=0.0,
    closed_position=2.18,
    open_width=0.050,
    width_per_radian=0.023,
    grasp_seed_arm_positions=(-1.25, -0.75, 0.5, 0.0, 0.0, 1.0, 0.0),
    transit_height=0.15,
    drop_slot_offsets=((-0.025, -0.027), (0.025, -0.027), (-0.025, 0.027), (0.025, 0.027)),
)

TRISKEL_CLEANUP_VIEW = CleanupViewConfig()

TRISKEL_CLEANUP_SCENE = CleanupSceneConfig(
    floor_height=-0.033,
    bin_position=(-0.55, 0.22),
    bin_half_size=(0.058, 0.062),
    bin_height=0.045,
    wall_thickness=0.006,
    cameras=TRISKEL_CLEANUP_CAMERAS,
    objects=(
        SceneObjectConfig(
            name="red_block",
            kind="block",
            position=(-0.50, 0.90),
            size=(0.028, 0.022, 0.026),
            color=(0.85, 0.12, 0.08, 1.0),
            yaw=0.2,
        ),
        SceneObjectConfig(
            name="yellow_block",
            kind="block",
            position=(0.02, 0.95),
            size=(0.024, 0.026, 0.024),
            color=(0.95, 0.65, 0.06, 1.0),
            yaw=-0.25,
        ),
        SceneObjectConfig(
            name="green_block",
            kind="block",
            position=(0.58, 0.95),
            size=(0.025, 0.020, 0.030),
            color=(0.12, 0.7, 0.18, 1.0),
            yaw=0.5,
        ),
        SceneObjectConfig(
            name="orange_car",
            kind="car",
            position=(0.45, 0.32),
            size=(0.042, 0.023, 0.022),
            color=(0.95, 0.3, 0.05, 1.0),
            yaw=-0.15,
        ),
    ),
)

TRISKEL_MOBILE_CLEANUP_CONFIG = MobileCleanupConfig(
    navigation_min=(-0.95, -0.50),
    navigation_max=(0.95, 1.20),
    pickup_offset=(0.0, 0.305),
)
