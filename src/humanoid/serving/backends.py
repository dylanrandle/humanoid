"""Lazy GPU backends. Imported only by their dedicated resident workers."""

# ruff: noqa: PLC0415 -- each model has its own CUDA dependency environment.

import time

import numpy as np

from humanoid.types.serving import ServingConfig


class GraspBackend:
    def __init__(self, config: ServingConfig):
        import os

        import torch
        from graspgenx import get_checkpoints_version_dir
        from graspgenx.models.grasp_gen import GraspGen
        from graspgenx.utils.checkpoint_io import load_model_cfg

        if not torch.cuda.is_available():
            raise RuntimeError("GraspGenX requires CUDA-enabled PyTorch")
        os.environ.setdefault("GRASPGENX_GRIPPER_CFG_DIR", config.model_directory)
        torch.set_num_threads(4)
        self.config = config
        checkpoints = get_checkpoints_version_dir()
        self.model_config = load_model_cfg(str(checkpoints / "gen"), str(checkpoints / "dis"))
        self.model = GraspGen.from_config(
            self.model_config.diffusion, self.model_config.discriminator
        )
        for component, checkpoint in (
            (self.model.grasp_generator, self.model_config.eval.gen_checkpoint),
            (self.model.grasp_discriminator, self.model_config.eval.dis_checkpoint),
        ):
            weights = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
            component.load_state_dict(weights["model"], strict=True, assign=True)
            del weights
        self.model = self.model.cuda().eval()
        self.conditioning: np.ndarray | None = None
        self.sampler = None
        # Initialize CUDA kernels/workspaces before reporting READY. Warmup output
        # is discarded; only camera-derived requests can enter the pickup planner.
        points = np.random.default_rng(0).uniform(-0.025, 0.025, (config.grasp_warmup_points, 3))
        self.predict(
            points.astype(np.float32),
            np.asarray(config.grasp_warmup_conditioning, dtype=np.float32),
        )

    def predict(self, points: np.ndarray, conditioning: np.ndarray) -> dict[str, np.ndarray]:
        import torch
        from graspgenx.grasp_server import GraspGenXSampler

        if (
            points.ndim != 2  # noqa: PLR2004 -- point cloud rank
            or points.shape[1:] != (3,)
            or len(points) < MIN_OBJECT_POINTS
            or conditioning.shape != (12,)
            or not np.isfinite(points).all()
            or not np.isfinite(conditioning).all()
        ):
            raise ValueError("Invalid grasp cloud or gripper conditioning")
        if (
            self.sampler is None
            or self.conditioning is None
            or not np.array_equal(self.conditioning, conditioning)
        ):
            self.sampler = GraspGenXSampler.from_sweep_volume(
                self.model_config,
                conditioning,
                model=self.model,
                use_fast_inference=False,
                obb_grasps=0,
            )
            self.conditioning = conditioning.copy()
        started = time.monotonic()
        with torch.inference_mode():
            poses, scores = GraspGenXSampler.run_inference(
                points,
                self.sampler,
                grasp_threshold=self.config.grasp_threshold,
                num_grasps=self.config.grasp_count,
                min_grasps=1,
                max_tries=1,
                remove_outliers=False,
            )
        torch.cuda.synchronize()
        print(f"GraspGenX resident inference: {time.monotonic() - started:.3f}s", flush=True)
        return {
            "poses": poses.detach().cpu().numpy().reshape(-1, 4, 4),
            "scores": scores.detach().cpu().numpy().reshape(-1),
        }


MIN_OBJECT_POINTS = 100
