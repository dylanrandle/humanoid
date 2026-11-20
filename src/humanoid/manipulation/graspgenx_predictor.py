"""GraspGenX diffusion adapter; inference runs in a separate CUDA environment."""

from collections.abc import Callable
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin

from humanoid.robots.base import Robot
from humanoid.types.pick import (
    CameraPointCloud,
    GraspGenXConfig,
    GraspInference,
    GraspProposal,
    PickConfig,
)

# Upstream: +Z approaches, +X closes. Triskel tool: +Y approaches, +X closes.
TOOL_FROM_NATIVE = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
NATIVE_FROM_CANONICAL = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
POSE_SHAPE = (4, 4)
ROTATION_TOLERANCE = 1e-4
MIN_OBJECT_POINT_COUNT = 100  # Released GraspGenX sampler contract.


def gripper_geometry(
    robot: Robot, config: GraspGenXConfig
) -> tuple[np.ndarray, list[tuple[coal.ConvexBase, pin.SE3]]]:
    """Inner-finger boxes at open/half-open and conservative open collision hulls.

    This describes the existing rigid gripper, not object-dependent grasp geometry.
    Finray deformation is deliberately not modeled.
    """
    q = pin.neutral(robot.model)
    data = robot.model.createData()
    geometry_data = pin.GeometryData(robot.visual_model)
    conditioning = []
    hulls = []
    low, high = robot.get_gripper_limits()[0]
    for state in (low, (low + high) / 2):
        robot.set_gripper_positions(q, np.array([state]))
        pin.updateGeometryPlacements(robot.model, data, robot.visual_model, geometry_data, q)
        native_from_world = (
            pin.SE3(TOOL_FROM_NATIVE.T, np.zeros(3)) * robot.get_tool_pose(q).inverse()
        )
        bounds = {}
        for geometry, world_pose in zip(
            robot.visual_model.geometryObjects, geometry_data.oMg, strict=True
        ):
            link = robot.model.frames[geometry.parentFrame].name
            if link not in config.gripper_links:
                continue
            if not isinstance(geometry.geometry, coal.BVHModelOBBRSS):
                raise ValueError(f"Expected a gripper mesh for {geometry.name}")
            vertices = np.asarray(geometry.geometry.vertices())
            native_pose = native_from_world * world_pose
            filename = Path(geometry.meshPath).name
            if filename in config.finger_meshes:
                points = vertices @ native_pose.rotation.T + native_pose.translation
                bounds[filename] = (points.min(axis=0), points.max(axis=0))
            if state == low:
                # Per-piece convex hulls conservatively fill the rigid fin's internal ribs.
                hull_vertices = coal.StdVec_Vec3s()
                hull_vertices.extend(vertices)
                hull = coal.Convex.convexHull(hull_vertices, True, "Qt")
                hulls.append((hull, native_pose.copy()))
        if set(bounds) != set(config.finger_meshes):
            raise ValueError("Missing configured Finray meshes in the robot visual model")
        left, right = (bounds[name] for name in config.finger_meshes)
        minimum = np.maximum(left[0], right[0])
        maximum = np.minimum(left[1], right[1])
        minimum[0], maximum[0] = left[1][0], right[0][0]
        extents = maximum - minimum
        if not np.all(extents > 0):
            raise ValueError("Gripper inner-finger bounds must have positive volume")
        conditioning.extend((extents, (maximum + minimum) / 2))
    if not hulls:
        raise ValueError("No gripper collision geometry")
    return np.concatenate(conditioning).astype(np.float32), hulls


def estimate_width(
    points: np.ndarray, pose: pin.SE3, conditioning: np.ndarray, config: PickConfig
) -> float | None:
    """Estimate the closing gap from observed object geometry at a learned pose.

    Like upstream's compute_close_value_for_object, project the object onto the
    closing axis. Use segmented depth instead of a privileged object mesh. Since
    our fingers close symmetrically, include offset from their URDF-derived center:
    a span alone could drive the nearer finger too far into an off-center object.
    Keep all target points (no clipping to the opening or percentile trimming).
    This estimates visible extent only; it cannot recover occluded surfaces.
    """
    if len(points) < config.minimum_points or not np.isfinite(points).all():
        return None
    native_points = (points - pose.translation) @ pose.rotation
    opening_center = conditioning[3]
    offsets = native_points[:, 0] - opening_center
    minimum, maximum = float(offsets.min()), float(offsets.max())
    if not minimum < 0 < maximum:
        return None
    width = 2 * max(-minimum, maximum)
    maximum_width = min(config.open_gap, float(conditioning[0]))
    return width if config.squeeze < width <= maximum_width else None


def convert_pose(
    pose: np.ndarray, score: float, config: PickConfig, width: float | None = None
) -> GraspProposal:
    """Map the predicted native gripper root to the shared fingertip convention."""
    rotation = pose[:3, :3]
    tip_native = TOOL_FROM_NATIVE.T @ np.asarray(config.tool_tip_offset)
    return GraspProposal(
        pose=pin.SE3(rotation @ NATIVE_FROM_CANONICAL, pose[:3, 3] + rotation @ tip_native),
        width=width,
        height=config.gripper_height,
        depth=0.0,
        score=score,
    )


class GraspGenXPredictor:
    def __init__(
        self,
        inference: GraspInference,
        robot: Robot,
        config: PickConfig,
        model_config: GraspGenXConfig,
    ) -> None:
        self.inference = inference
        self.config = config
        self.model_config = model_config
        self.conditioning, self.hulls = gripper_geometry(robot, model_config)

    def _infer(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.inference.grasp(points, self.conditioning)

    def _clear_approach(self, pose: pin.SE3, scene: coal.OcTree) -> bool:
        steps = int(np.ceil(self.config.approach_clearance / self.model_config.approach_step))
        request = coal.CollisionRequest()
        for distance in np.linspace(0, self.config.approach_clearance, steps + 1):
            approach = pose.copy()
            approach.translation -= pose.rotation[:, 2] * distance
            for hull, local in self.hulls:
                placed = approach * local
                if coal.collide(
                    hull,
                    coal.Transform3s(placed.rotation, placed.translation),
                    scene,
                    coal.Transform3s(),
                    request,
                    coal.CollisionResult(),
                ):
                    return False
        return True

    def predict(
        self,
        cloud: CameraPointCloud,
        region: np.ndarray,
        on_candidates: Callable[[tuple[GraspProposal, ...]], None] | None = None,
    ) -> list[GraspProposal]:
        if region.shape != (len(cloud.points),) or region.dtype != np.bool_:
            raise ValueError("GraspGenX region must be a boolean mask matching the depth cloud")
        if np.count_nonzero(region) < max(self.config.minimum_points, MIN_OBJECT_POINT_COUNT):
            raise ValueError("Refusing GraspGenX inference without a populated target region")
        if not np.isfinite(cloud.points).all():
            raise ValueError("GraspGenX point cloud contains non-finite coordinates")
        object_points = np.ascontiguousarray(cloud.points[region], dtype=np.float32)
        poses, scores = self._infer(object_points)
        if scores.ndim != 1 or poses.shape != (len(scores), *POSE_SHAPE):
            raise ValueError("Unexpected GraspGenX output; expected N x 4 x 4 poses and N scores")
        generated = []
        for index in np.argsort(-scores):
            matrix, score = poses[index], float(scores[index])
            rotation = matrix[:3, :3]
            if (
                not np.isfinite(matrix).all()
                or not np.isfinite(score)
                or score < self.model_config.minimum_score
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=ROTATION_TOLERANCE)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=ROTATION_TOLERANCE)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=ROTATION_TOLERANCE)
            ):
                continue
            pose = pin.SE3(rotation.copy(), matrix[:3, 3].copy())
            width = estimate_width(object_points, pose, self.conditioning, self.config)
            generated.append((matrix, score, width))
        if on_candidates is not None:
            on_candidates(
                tuple(
                    convert_pose(matrix, score, self.config, width)
                    for matrix, score, width in generated[: self.config.max_grasp_candidates]
                )
            )
        if not generated:
            return []
        # Retain unsegmented and RGB-invisible depth points as obstacles.
        scene = coal.makeOctree(cloud.points.astype(float), self.model_config.collision_resolution)
        proposals = []
        for matrix, score, width in generated:
            if width is None:
                continue
            pose = pin.SE3(matrix[:3, :3].copy(), matrix[:3, 3].copy())
            if self._clear_approach(pose, scene):
                proposals.append(convert_pose(matrix, score, self.config, width))
                if len(proposals) == self.config.max_grasp_candidates:
                    break
        return proposals
