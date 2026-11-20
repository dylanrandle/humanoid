"""Resident licensed AnyGrasp detector for an entire camera-frame point cloud."""

import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from humanoid.types.serving import AnyGraspConfig, Arrays


class AnyGraspBackend:
    def __init__(self, config: AnyGraspConfig):
        self.config = config
        sdk = Path(config.sdk_directory).expanduser().resolve()
        if not (sdk / "gsnet.so").is_file():
            raise RuntimeError("Install the official AnyGrasp binary for this Python/architecture")
        # The SDK resolves its license relative to cwd. This is an isolated worker,
        # so changing directory cannot affect the control or other model processes.
        os.chdir(sdk)
        sys.path.insert(0, str(sdk))
        from gsnet import check_license, create_detector  # noqa: PLC0415

        if not check_license("license"):
            raise RuntimeError("AnyGrasp license rejected; check the SDK's machine-ID diagnostic")
        checkpoint = Path(config.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f"AnyGrasp checkpoint is missing: {checkpoint}")
        import torch  # noqa: PLC0415 -- licensed CUDA worker only

        if not torch.cuda.is_available():
            raise RuntimeError("AnyGrasp requires CUDA-enabled PyTorch")
        torch.set_num_threads(2)
        self.torch = torch
        self.detector = create_detector(
            Namespace(
                checkpoint_path=str(checkpoint),
                max_gripper_width=config.max_gripper_width,
                gripper_height=config.gripper_height,
            )
        )
        if self.detector is None:
            raise RuntimeError("AnyGrasp detector initialization failed")
        # Discard warmup predictions. Actual requests only use observed depth.
        points = np.random.default_rng(0).uniform(-0.03, 0.03, (config.warmup_points, 3))
        points[:, 2] += 0.4
        self.predict(points.astype(np.float32))

    def predict(self, points: np.ndarray) -> Arrays:
        config = self.config
        if (
            points.ndim != 2  # noqa: PLR2004 -- Nx3 point cloud
            or points.shape[1:] != (3,)
            or not config.minimum_points <= len(points) <= config.maximum_points
            or not np.isfinite(points).all()
            or np.any(points[:, 2] <= 0)
        ):
            raise ValueError("AnyGrasp expects a finite camera-frame Nx3 point cloud in meters")
        with self.torch.inference_mode():
            group = self.detector.get_grasp(
                np.ascontiguousarray(points, dtype=np.float32),
                {"dense_grasp": False, "collision_detection": True},
            )
        self.torch.cuda.synchronize()
        if group is None or len(group) == 0:
            return {
                "poses": np.empty((0, 4, 4), dtype=np.float32),
                **{
                    key: np.empty(0, dtype=np.float32)
                    for key in ("scores", "widths", "heights", "depths")
                },
            }
        # No region steering, semantic labels, external segmentation or prompt.
        group = group.nms().sort_by_score()[: config.maximum_grasps]
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(group), axis=0)
        poses[:, :3, :3] = group.rotation_matrices
        poses[:, :3, 3] = group.translations
        result = {"poses": poses}
        for key in ("scores", "widths", "heights", "depths"):
            result[key] = np.asarray(getattr(group, key), dtype=np.float32)
        if any(not np.isfinite(array).all() for array in result.values()):
            raise RuntimeError("AnyGrasp returned non-finite grasp parameters")
        keep = (
            (result["scores"] >= config.minimum_score)
            & (result["widths"] > 0)
            & (result["widths"] <= config.max_gripper_width)
            & (result["heights"] > 0)
            & (result["depths"] >= 0)
        )
        return {key: value[keep] for key, value in result.items()}
