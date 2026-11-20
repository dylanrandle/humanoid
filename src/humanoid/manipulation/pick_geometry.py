"""Register native RGB/depth and convert learned grasps into Triskel targets."""

import numpy as np
import pinocchio as pin

from humanoid.robots.base import Robot
from humanoid.types.camera import CameraFrames, CameraIntrinsics
from humanoid.types.pick import (
    CameraPointCloud,
    GraspProposal,
    ObjectMask,
    PickConfig,
    PickPlan,
    PickWaypoint,
)

BROWN_COEFFICIENT_COUNT = 5
MIN_CAMERA_DEPTH = 0.05
MAX_CAMERA_DEPTH = 1.5


def camera_pose(robot: Robot, q: np.ndarray, frame: str) -> pin.SE3:
    world_camera = robot.get_frame_pose(frame, q).copy()
    world_base = robot.get_base_pose(q)
    return world_camera if world_base is None else world_base.inverse() * world_camera


def _brown(points: np.ndarray, coefficients: tuple[float, ...]) -> np.ndarray:
    x, y = points.T
    k1, k2, p1, p2, k3 = coefficients
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    return np.column_stack(
        (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y * y),
        )
    )


def distort(points: np.ndarray, intrinsics: CameraIntrinsics, *, inverse: bool) -> np.ndarray:
    """Brown-Conrady normalized coordinates; refuse unknown nonzero calibration."""
    coefficients = intrinsics.distortion_coefficients
    if not any(coefficients):
        return points.copy()
    if (
        intrinsics.distortion_model != "brown_conrady"
        or len(coefficients) != BROWN_COEFFICIENT_COUNT
    ):
        raise ValueError(f"Unsupported nonzero distortion: {intrinsics.distortion_model}")
    if not inverse:
        return _brown(points, coefficients)
    estimate = points.copy()
    for _ in range(20):
        estimate += points - _brown(estimate, coefficients)
    if not np.allclose(_brown(estimate, coefficients), points, atol=1e-6):
        raise ValueError("Depth distortion inversion did not converge")
    return estimate


def reconstruct_scene(frames: CameraFrames, robot: Robot, q: np.ndarray) -> CameraPointCloud:
    """Keep the full depth cloud for collisions; associate only RGB-visible samples.

    Depth is optical Z, not ray distance. Occluded/out-of-view samples retain
    (-1, -1) RGB indices and cannot acquire a segmentation label.
    """
    rgb, depth = frames.rgb, frames.depth
    if rgb.timestamp != depth.timestamp or rgb.serial_number != depth.serial_number:
        raise ValueError("RGB and depth must belong to the same camera frameset")
    base_depth = camera_pose(robot, q, depth.frame_id)
    base_rgb = camera_pose(robot, q, rgb.frame_id)
    rgb_depth = base_rgb.inverse() * base_depth
    rows, columns = np.nonzero(depth.data)
    z = depth.data[rows, columns].astype(float) * depth.depth_scale
    intr = depth.intrinsics
    rays = distort(
        np.column_stack(((columns - intr.cx) / intr.fx, (rows - intr.cy) / intr.fy)),
        intr,
        inverse=True,
    )
    points = np.column_stack((rays * z[:, None], z))
    points = points[(z > MIN_CAMERA_DEPTH) & (z < MAX_CAMERA_DEPTH)]
    pixels = np.full((len(points), 2), -1, dtype=np.int64)
    color = points @ rgb_depth.rotation.T + rgb_depth.translation
    indices = np.flatnonzero(color[:, 2] > MIN_CAMERA_DEPTH)
    color = color[indices]
    intr = rgb.intrinsics
    xy = distort(color[:, :2] / color[:, 2, None], intr, inverse=False)
    u = np.rint(intr.fx * xy[:, 0] + intr.cx).astype(int)
    v = np.rint(intr.fy * xy[:, 1] + intr.cy).astype(int)
    inside = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    indices, color, u, v = indices[inside], color[inside], u[inside], v[inside]
    nearest = np.full((intr.height, intr.width), np.inf)
    np.minimum.at(nearest, (v, u), color[:, 2])
    visible = color[:, 2] <= nearest[v, u] + 0.002
    pixels[indices[visible]] = np.column_stack((v[visible], u[visible]))
    return CameraPointCloud(
        points=points.astype(np.float32), rgb_pixels=pixels, base_from_camera=base_depth
    )


def object_region(mask: ObjectMask, cloud: CameraPointCloud, config: PickConfig) -> np.ndarray:
    """Target mask indexed exactly like the full native depth cloud."""
    pixels = cloud.rgb_pixels
    visible = (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0)
    selected = np.zeros(len(cloud.points), dtype=bool)
    selected[visible] = mask.mask[pixels[visible, 0], pixels[visible, 1]]
    if np.count_nonzero(selected) < config.minimum_points:
        # Require observed object geometry before asking for grasps.
        raise ValueError("Too few segmented depth samples for grasp inference")
    return selected


def plan_pick(
    grasp: GraspProposal,
    cloud: CameraPointCloud,
    region: np.ndarray,
    config: PickConfig,
    robot: Robot,
) -> PickPlan:
    """Preserve learned orientation/insertion depth and map the canonical grasp axes to the fins."""
    if grasp.width is not None and not config.squeeze < grasp.width <= config.open_gap:
        raise ValueError("Grasp width cannot fit the opening and squeeze preload")
    base_grasp = cloud.base_from_camera * grasp.pose
    approach = base_grasp.rotation[:, 0]
    # Canonical +X = approach, +Y = opening. Triskel +Y = fins, +X = opening.
    # Flip the third axis as well to preserve a proper rotation (determinant +1).
    grasp_from_tool = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    rotation = base_grasp.rotation @ grasp_from_tool
    tip = base_grasp.translation + grasp.depth * approach
    points = cloud.base_points[region]
    if len(points) < config.minimum_points:
        raise ValueError("No supported target object for this grasp")
    position = tip - rotation @ np.asarray(config.tool_tip_offset)
    pose = pin.SE3(rotation, position)
    limits = robot.get_gripper_limits()
    if len(limits) != 1:
        raise ValueError("Pickup requires one gripper actuator")
    phases = [
        ("approach", -approach * config.approach_clearance, 0.0, 0.0),
        ("grasp", np.zeros(3), 0.0, 0.0),
    ]
    # Pose-only models provide an approach preview, never an invented closure target.
    if grasp.width is not None:
        closed = (config.open_gap - (grasp.width - config.squeeze)) / config.gap_per_radian
        if not limits[0][0] <= closed <= limits[0][1]:
            raise ValueError("Requested squeeze exceeds the configured gripper limits")
        phases.extend(
            [
                ("squeeze", np.zeros(3), closed, config.squeeze_seconds),
                ("lift", np.array([0, 0, config.lift_height]), closed, 1.0),
            ]
        )
    waypoints = []
    for name, offset, gripper, hold in phases:
        target = pose.copy()
        target.translation += offset
        if np.any(target.translation < config.workspace_min) or np.any(
            target.translation > config.workspace_max
        ):
            raise ValueError(f"{name} tool target is outside the pickup workspace")
        waypoints.append(PickWaypoint(name=name, pose=target, gripper=gripper, hold_seconds=hold))
    return PickPlan(grasp=grasp, object_points=points, waypoints=tuple(waypoints))
