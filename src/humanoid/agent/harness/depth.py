"""On-demand depth completion for the current RGB-D observation."""

from contextlib import ExitStack

import numpy as np
from PIL import Image

from humanoid.agent.harness.observation import PinnedObservation
from humanoid.config.camera import CAMERA_PREVIEW_CONFIG
from humanoid.perception.depth import DepthCompleter, colorize_depth
from humanoid.types.depth import DepthCamera
from humanoid.types.tools import CompleteDepth, DepthCompletionResult


class DepthTools:
    def __init__(self, observation: PinnedObservation):
        self.observation = observation
        self.resources = ExitStack()
        self.depth_models: dict[DepthCamera, DepthCompleter] = {}

    def close(self):
        self.resources.close()

    def complete_depth(self, request: CompleteDepth) -> tuple[DepthCompletionResult, Image.Image]:
        observation_id = request.observation_id
        frames = self.observation.frames
        if frames is None or observation_id != self.observation.id:
            raise ValueError("Unknown observation ID for this view.")
        reg = frames.registered
        if reg is None:
            raise ValueError(
                "This observation has no synchronized RGB-aligned depth. Request a new observation."
            )
        self.observation.save()
        depth = self.observation.store.resolve_depth(observation_id)
        if depth is None:
            if reg.camera_model not in self.depth_models:
                self.depth_models[reg.camera_model] = self.resources.enter_context(
                    DepthCompleter(reg.camera_model)
                )
            depth = self.depth_models[reg.camera_model].infer(
                frames.rgb.data, reg.image.data, metres_per_unit=reg.image.depth_scale
            )
            self.observation.store.save_depth(observation_id, depth)
        near = CAMERA_PREVIEW_CONFIG.depth_near_meters
        far = CAMERA_PREVIEW_CONFIG.default_depth_far_meters
        return DepthCompletionResult(
            observation_id=observation_id,
            valid_pixels=int(np.count_nonzero(np.isfinite(depth) & (depth > 0))),
            preview_range_m=(near, far),
        ), Image.fromarray(colorize_depth(depth, near, far))
