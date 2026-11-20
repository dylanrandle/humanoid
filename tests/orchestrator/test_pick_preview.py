"""Annotated camera frames traverse the private policy API separately from status."""

from dataclasses import asdict
from unittest.mock import MagicMock

from humanoid.nodes.policy.autonomy.pick import PickNode
from humanoid.orchestrator.pick import PickClient
from humanoid.serving.transport import json_arrays, read_json
from humanoid.types.camera import CameraInferencePreview, CameraPreview


def test_policy_preview_serializes_exact_source_frames_and_empty_results():
    node = PickNode.__new__(PickNode)
    node.policy = MagicMock()
    preview = CameraInferencePreview(
        rgb=CameraPreview(connected=True, image="rgb", frame_id="rgb", frame_number=1),
        depth=CameraPreview(connected=True, image="depth", frame_id="depth", frame_number=2),
        object_description="toy car",
        mask_count=1,
        grasp_count=3,
        selected=True,
    )
    node.policy.preview.return_value = preview
    result = node._handle(json_arrays({"operation": "preview", "depth_far": 3.0}))
    assert read_json(result) == asdict(preview)
    node.policy.preview.assert_called_once_with(3.0)
    node.policy.preview.return_value = None
    assert read_json(node._handle(json_arrays({"operation": "preview", "depth_far": 3}))) is None


def test_preview_client_reconstructs_nested_types_and_drops_disconnected_results(monkeypatch):
    response = {
        "rgb": {"connected": True, "image": "rgb", "frame_number": 1},
        "depth": {"connected": True, "image": "depth", "frame_number": 2},
        "object_description": "toy car",
        "mask_count": 1,
        "grasp_count": 2,
        "selected": False,
    }
    request = MagicMock(return_value=json_arrays(response))
    monkeypatch.setattr("humanoid.orchestrator.pick.request", request)
    client = PickClient("unused.sock")
    preview = client.preview(5)
    assert isinstance(preview, CameraInferencePreview)
    assert isinstance(preview.rgb, CameraPreview)
    assert preview.rgb.frame_number != preview.depth.frame_number
    assert read_json(request.call_args.args[1]) == {"operation": "preview", "depth_far": 5}
    request.side_effect = ConnectionError("stopped")
    assert client.preview(5) is None
