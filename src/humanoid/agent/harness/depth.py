"""On-demand depth completion for the current RGB-D observation."""

from contextlib import ExitStack

import numpy as np

from humanoid.agent.harness.observation import PinnedObservation
from humanoid.perception.depth import DepthCompleter
from humanoid.types.camera import DepthCamera
from humanoid.types.tools import CompleteDepth, DepthCompletionResult, DepthSamples, ReadDepth


class DepthTools:
    def __init__(self, observation: PinnedObservation):
        self.observation = observation
        self.resources = ExitStack()
        self.depth_models: dict[DepthCamera, DepthCompleter] = {}

    def close(self):
        self.resources.close()

    def read_depth(self, request: ReadDepth) -> DepthSamples:
        self.observation.select(request.observation_id)
        frames = self.observation.frames
        assert frames is not None
        frame = frames.depth_image(request.view)
        if frame is None:
            raise ValueError(f"This observation has no {request.view} depth.")
        height, width = frame.data.shape
        depths = []
        for x, y in request.points_xy:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Depth coordinates must be inside the {width} by {height} image.")
            sample = int(frame.data[y, x])
            depths.append(sample * frame.depth_scale if sample else None)
        return DepthSamples(
            observation_id=request.observation_id, view=request.view, depths_m=tuple(depths)
        )

    def complete_depth(self, request: CompleteDepth) -> DepthCompletionResult:
        observation_id = request.observation_id
        self.observation.select(observation_id)
        frames = self.observation.frames
        if frames is None or observation_id != self.observation.id:
            raise ValueError("Unknown observation ID for this view.")
        reg = frames.registered
        if reg is None:
            raise ValueError(
                "This observation has no synchronized RGB-aligned depth. Request a new observation."
            )
        height, width = reg.image.data.shape
        if any(not (0 <= x < width and 0 <= y < height) for x, y in request.points_xy):
            raise ValueError(f"Depth coordinates must be inside the {width} by {height} RGB image.")
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
        samples = [float(depth[y, x]) for x, y in request.points_xy]
        return DepthCompletionResult(
            observation_id=observation_id,
            valid_pixels=int(np.count_nonzero(np.isfinite(depth) & (depth > 0))),
            depths_m=tuple(v if np.isfinite(v) and v > 0 else None for v in samples),
        )
