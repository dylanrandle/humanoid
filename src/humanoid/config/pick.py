"""Triskel pickup defaults; distances are meters in the tool-command/base frame."""

from pathlib import Path

from humanoid.types.pick import GraspGenXConfig, PickConfig

GRASPGENX_CONFIG = GraspGenXConfig(
    finger_meshes=("finger_2_fin.stl", "finger_1_fin.stl"),
    gripper_links=(
        "gripper_base_link",
        "finger_drive_link",
        "gripper_gear_link",
        "finger_1_link",
        "finger_2_link",
    ),
    num_grasps=100,
    minimum_score=0.7,
    collision_resolution=0.002,
    approach_step=0.002,
)

PICK_CONFIG = PickConfig(
    segmenter_model="openvision/yoloe26-s-seg",
    segmenter_revision="86f1684b97c4f14ae1434be945be7aa172fd6987",
    segmentation_image_size=640,
    detection_threshold=0.3,
    # Calibrated Triskel tool-to-fingertip offset for the grasp frame conversion.
    tool_tip_offset=(0.002, 0.134, 0.0),
    gripper_height=0.040,
    max_grasp_candidates=20,
    open_gap=0.050,
    gap_per_radian=0.023,  # Two 11.5 mm/rad URDF mimic slides.
    squeeze=0.004,
    approach_clearance=0.060,
    lift_height=0.080,
    workspace_min=(-0.16, 0.20, -0.032),
    workspace_max=(0.16, 0.40, 0.30),
    linear_speed=0.025,
    angular_speed=0.35,
    gripper_speed=0.5,
    rate_hz=30.0,
    phase_timeout=25.0,
    feedback_timeout=0.75,
    image_max_age=0.3,
    position_tolerance=0.007,
    orientation_tolerance=0.10,
    squeeze_seconds=1.5,
    minimum_points=30,
)

# Local control socket; browser requests continue through the existing HTTP UI.
PICK_CONTROL_SOCKET = str(Path.home() / ".cache/humanoid/pick.sock")
