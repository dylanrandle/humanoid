"""Camera perception and kinematic planning outside the realtime policy tick."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from humanoid.config.camera import REALSENSE_CAMERA_CONFIG
from humanoid.manipulation.graspgenx_predictor import GraspGenXPredictor
from humanoid.manipulation.pick_geometry import (
    camera_pose,
    object_region,
    plan_pick,
    reconstruct_scene,
)
from humanoid.manipulation.pick_motion import check_reachability
from humanoid.robots.base import Robot
from humanoid.serving.client import ModelClient
from humanoid.types.observation import Observation
from humanoid.types.pick import (
    GraspGenXConfig,
    GraspProposal,
    ObjectMask,
    PickConfig,
    PickPerceptionPreview,
    PickPlan,
)

MAX_JOINT_DRIFT = 0.02
MAX_SCENE_SHIFT = 0.005
MAX_RGB_CHANGE = 20


def perceive(  # noqa: PLR0913 -- dependencies are explicit and testable
    observation: Observation,
    description: str,
    robot: Robot,
    config: PickConfig,
    grasp_config: GraspGenXConfig,
    models: ModelClient,
    on_preview: Callable[[PickPerceptionPreview], None] | None = None,
) -> tuple[PickPlan, ObjectMask]:
    frames = observation.camera_frames
    if frames is None:
        raise ValueError("Waiting for paired RGB and depth frames")
    if (frames.rgb.frame_id, frames.depth.frame_id) != (
        REALSENSE_CAMERA_CONFIG.rgb_frame,
        REALSENSE_CAMERA_CONFIG.depth_frame,
    ):
        raise ValueError("Expected Triskel RGB/depth optical frames")
    cloud = reconstruct_scene(frames, robot, observation.robot_state.joint_positions)
    masks = sorted(
        models.segment(frames.rgb.data, description), key=lambda m: m.score, reverse=True
    )
    preview = PickPerceptionPreview(
        frames=frames,
        cloud=cloud,
        rgb_from_depth=camera_pose(
            robot, observation.robot_state.joint_positions, frames.rgb.frame_id
        ).inverse()
        * cloud.base_from_camera,
        object_description=description,
        masks=tuple(masks),
    )
    if on_preview is not None:
        on_preview(preview)

    def candidates_ready(grasps: tuple[GraspProposal, ...]) -> None:
        nonlocal preview
        preview = replace(preview, grasps=grasps)
        if on_preview is not None:
            on_preview(preview)

    if not masks:
        raise RuntimeError(f'Could not find "{description}" in the camera image')
    predictor = GraspGenXPredictor(models, robot, config, grasp_config)
    reasons = []
    for mask in masks:
        try:
            region = object_region(mask, cloud, config)
            candidates = predictor.predict(cloud, region, candidates_ready)
            for grasp in candidates:
                try:
                    plan = plan_pick(grasp, cloud, region, config, robot)
                    check_reachability(plan, robot, observation.robot_state.joint_positions, config)
                    if on_preview is not None:
                        on_preview(replace(preview, selected=grasp))
                    return plan, mask
                except (RuntimeError, ValueError) as error:
                    reasons.append(str(error))
        except ValueError as error:
            reasons.append(str(error))
    detail = "; ".join(reasons[-3:]) or "No supported, collision-free reachable grasp"
    raise RuntimeError(detail)


def confirm_scene(  # noqa: PLR0913 -- compares an observation/plan pair with current RGB-D
    original: Observation,
    fresh: Observation,
    plan: PickPlan,
    mask: ObjectMask,
    robot: Robot,
    config: PickConfig,
) -> None:
    """Recheck depth and appearance immediately before starting the approach."""
    q = fresh.robot_state.joint_positions
    if np.max(np.abs(q - original.robot_state.joint_positions)) > MAX_JOINT_DRIFT:
        raise RuntimeError("Robot moved during perception/preflight; acquire a new grasp")
    assert fresh.camera_frames is not None and original.camera_frames is not None
    cloud = reconstruct_scene(fresh.camera_frames, robot, q)
    region = object_region(mask, cloud, config)
    latest_points = cloud.base_points[region]
    image_change = np.mean(
        np.abs(
            fresh.camera_frames.rgb.data[mask.mask].astype(float)
            - original.camera_frames.rgb.data[mask.mask]
        )
    )
    # Compare the segmented cloud, not a size/yaw-based grasp heuristic.
    reference = np.quantile(plan.object_points, [0.1, 0.5, 0.9], axis=0)
    latest = np.quantile(latest_points, [0.1, 0.5, 0.9], axis=0)
    if np.max(np.abs(latest - reference)) > MAX_SCENE_SHIFT or image_change > MAX_RGB_CHANGE:
        raise RuntimeError("Object/scene changed during perception/preflight; acquire a new grasp")
