"""Operator request client for the managed pickup policy process."""

from humanoid.serving.transport import json_arrays, read_json, request
from humanoid.types.camera import CameraInferencePreview, CameraPreview
from humanoid.types.pick import PickPhase, PickStatus


class PickClient:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path

    def status(self) -> PickStatus:
        try:
            result = read_json(request(self.socket_path, json_arrays({"operation": "status"}), 0.5))
            result["phase"] = PickPhase(result["phase"])
            return PickStatus(**result)
        except (OSError, ValueError, RuntimeError):
            return PickStatus(phase=PickPhase.READY, message="Start the pick policy to load models")

    def run(self, description: str) -> None:
        request(self.socket_path, json_arrays({"operation": "run", "object": description}), 1.0)

    def preview(self, depth_far: float) -> CameraInferencePreview | None:
        try:
            result = read_json(
                request(
                    self.socket_path,
                    json_arrays({"operation": "preview", "depth_far": depth_far}),
                    0.5,
                )
            )
            if result is None:
                return None
            return CameraInferencePreview(
                rgb=CameraPreview(**result["rgb"]),
                depth=CameraPreview(**result["depth"]),
                object_description=result["object_description"],
                mask_count=result["mask_count"],
                grasp_count=result["grasp_count"],
                selected=result["selected"],
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            return None

    def cancel(self) -> None:
        request(self.socket_path, json_arrays({"operation": "cancel"}), 1.0)
