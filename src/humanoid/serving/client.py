"""Lightweight client; policy processes never import a GPU framework."""

import numpy as np
import pinocchio as pin

from humanoid.serving.transport import read_json, request
from humanoid.types.pick import GraspProposal, ObjectMask
from humanoid.types.serving import ModelKind, ServingConfig, ServingState, ServingStatus


class ModelClient:
    def __init__(self, config: ServingConfig):
        self.config = config

    def status(self) -> ServingStatus:
        try:
            result = read_json(
                request(self.config.socket_path, {"operation": np.asarray("status")}, 0.5)
            )
            result["state"] = ServingState(result["state"])
            return ServingStatus(**result)
        except (OSError, RuntimeError, ValueError) as error:
            return ServingStatus(state=ServingState.STOPPED, message=f"Models unavailable: {error}")

    def segment(self, rgb: np.ndarray, prompt: str) -> list[ObjectMask]:
        result = request(
            self.config.socket_path,
            {
                "operation": np.asarray(ModelKind.SEGMENTATION),
                "rgb": rgb,
                "prompt": np.asarray(prompt),
            },
            self.config.request_timeout,
        )
        masks, scores, boxes = result["masks"], result["scores"], result["boxes"]
        if (
            scores.ndim != 1
            or masks.shape != (len(scores), *rgb.shape[:2])
            or masks.dtype != np.bool_
            or boxes.shape != (len(scores), 4)
            or not np.isfinite(scores).all()
            or not np.isfinite(boxes).all()
        ):
            raise ValueError("Unexpected segmentation output")
        return [
            ObjectMask(
                mask=mask,
                score=float(score),
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
            for mask, score, box in zip(masks, scores, boxes, strict=True)
        ]

    def grasp(self, points: np.ndarray, conditioning: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = request(
            self.config.socket_path,
            {
                "operation": np.asarray(ModelKind.GRASP),
                "points": points,
                "conditioning": conditioning,
            },
            self.config.request_timeout,
        )
        return result["poses"], result["scores"]

    def scene_grasps(self, points: np.ndarray) -> list[GraspProposal]:
        """AnyGrasp predictions in native depth optical coordinates, meters."""
        result = request(
            self.config.socket_path,
            {"operation": np.asarray(ModelKind.ANYGRASP), "points": points},
            self.config.request_timeout,
        )
        poses, scores = result["poses"], result["scores"]
        sizes = (result["widths"], result["heights"], result["depths"])
        if (
            scores.ndim != 1
            or poses.shape != (len(scores), 4, 4)
            or any(array.shape != scores.shape for array in sizes)
            or any(not np.isfinite(array).all() for array in (poses, scores, *sizes))
        ):
            raise ValueError("Malformed AnyGrasp output")
        rotations = poses[:, :3, :3]
        valid_rotations = np.allclose(
            rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4
        ) and np.allclose(np.linalg.det(rotations), 1, atol=1e-4)
        valid_sizes = (
            np.all(sizes[0] > 0)
            and np.all(sizes[0] <= self.config.anygrasp.max_gripper_width)
            and np.all(sizes[1] > 0)
            and np.all(sizes[2] >= 0)
        )
        if not valid_rotations or not valid_sizes or not np.allclose(poses[:, 3], [0, 0, 0, 1]):
            raise ValueError("Invalid AnyGrasp poses or dimensions")
        return [
            GraspProposal(
                pose=pin.SE3(pose[:3, :3], pose[:3, 3]),
                score=float(score),
                width=float(width),
                height=float(height),
                depth=float(depth),
            )
            for pose, score, width, height, depth in zip(poses, scores, *sizes, strict=True)
        ]
