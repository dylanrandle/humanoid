import json
import time
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest

from humanoid.agent.client import AgentControlClient
from humanoid.agent.harness import depth as depth_tools
from humanoid.agent.harness import segmentation
from humanoid.agent.harness.observation import PinnedObservation
from humanoid.agent.harness.robot import TriskelTools
from humanoid.agent.serialization import decode_frames, encode_frames
from humanoid.agent.server import AgentControlServer
from humanoid.config.camera import REALSENSE_CAMERA_CONFIG
from humanoid.perception.masks import MaskStore
from humanoid.types.agent import AgentControlEndpoint
from humanoid.types.camera import (
    CameraEncoding,
    CameraFrames,
    CameraImage,
    CameraIntrinsics,
    RegisteredDepth,
)
from humanoid.types.depth import DepthCamera
from humanoid.types.tools import (
    CompleteDepth,
    RefineMask,
    SegmentAtPoint,
    SegmentEverything,
    SegmentInBox,
)


@pytest.fixture
def frames():
    intr = CameraIntrinsics(
        width=32,
        height=24,
        fx=32,
        fy=33,
        cx=15,
        cy=11,
        distortion_model="none",
        distortion_coefficients=(),
    )
    rgb = CameraImage(
        timestamp=10,
        device_timestamp=1,
        timestamp_domain="device",
        frame_number=11,
        frame_id=REALSENSE_CAMERA_CONFIG.rgb_frame,
        serial_number="abc",
        encoding=CameraEncoding.RGB8,
        data=np.full((24, 32, 3), 15, np.uint8),
        intrinsics=intr,
    )
    native = replace(
        rgb,
        frame_number=12,
        frame_id="depth",
        encoding=CameraEncoding.DEPTH16,
        data=np.full((24, 32), 700, np.uint16),
        depth_scale=0.001,
    )
    aligned = replace(native, frame_id=rgb.frame_id, data=np.full((24, 32), 900, np.uint16))
    return CameraFrames(
        rgb=rgb,
        depth=native,
        registered=RegisteredDepth(
            camera_model=DepthCamera.D405, rgb_frame_number=rgb.frame_number, image=aligned
        ),
    )


def test_registered_depth_round_trip_rejects_other_capture(frames):
    restored = decode_frames(encode_frames(frames), received_after=100)
    assert restored.registered is not None
    assert restored.registered.camera_model is DepthCamera.D405
    np.testing.assert_array_equal(restored.registered.image.data, frames.registered.image.data)
    assert restored.registered.image.intrinsics == frames.rgb.intrinsics
    assert "image" not in encode_frames(frames)["registered"]
    assert "registered" not in encode_frames(frames, depth_view="native")
    with pytest.raises(ValueError, match="capture"):
        replace(frames, depth=replace(frames.depth, frame_number=13))
    with pytest.raises(ValueError, match="capture"):
        replace(frames, registered=replace(frames.registered, rgb_frame_number=10))


def test_default_view_preserves_rgb_when_aligned_depth_is_missing(frames):
    payload = encode_frames(replace(frames, registered=None))
    assert payload["depth_view"] == "aligned"
    assert payload["depth"] is None and "registered" not in payload
    view = decode_frames(payload, received_after=100)
    np.testing.assert_array_equal(view.rgb.data, frames.rgb.data)
    assert view.depth is None and view.registered is None


def test_http_pins_complete_capture_when_newer_native_pair_arrives(service, frames):
    now = time.perf_counter()
    old = replace(
        frames,
        rgb=replace(frames.rgb, timestamp=now),
        depth=replace(frames.depth, timestamp=now),
        registered=replace(
            frames.registered, image=replace(frames.registered.image, timestamp=now)
        ),
    )
    observation = service.observation()
    service.observe(replace(observation, camera_frames=old))
    new = replace(old, registered=None, rgb=replace(old.rgb, frame_number=old.rgb.frame_number + 1))
    service.observe(replace(observation, camera_frames=new))
    server = AgentControlServer(service)
    server.start()
    host, port = server.server.server_address[:2]
    client = AgentControlClient(AgentControlEndpoint(f"http://{host}:{port}"))
    try:
        native = client.observe(depth_view="native")
        assert native.rgb.frame_number == new.rgb.frame_number and native.depth is not None
        assert native.depth.frame_id == frames.depth.frame_id and native.registered is None
        np.testing.assert_array_equal(native.depth.data, frames.depth.data)
        pinned = client.observe()
        assert pinned.registered is not None and pinned.rgb.frame_number == old.rgb.frame_number
        assert pinned.depth is pinned.registered.image
        assert pinned.depth.intrinsics == pinned.rgb.intrinsics
        np.testing.assert_array_equal(pinned.depth.data, frames.registered.image.data)
        aligned = client._request("/v1/camera?depth_view=aligned")
        assert aligned["depth_view"] == "aligned"
        assert aligned["rgb"]["frame_number"] == pinned.rgb.frame_number
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize("aligned", [True, False])
def test_agent_observes_measured_depth_if_available(service, frames, aligned):
    client = MagicMock()
    client.observation.return_value = service.observation()
    source = frames if aligned else replace(frames, registered=None)
    payload = encode_frames(source)
    if aligned:
        assert "image" not in payload["registered"]
    else:
        assert payload["depth"] is None and "registered" not in payload
    client.observe.return_value = decode_frames(payload, received_after=100)
    tools = TriskelTools(client)
    try:
        status, rgb, depth = tools.observe(service.observation().mode)
        client.observe.assert_called_once_with()
        assert status["registered_depth_available"] is aligned
        assert rgb.startswith("data:image/jpeg;")
        if aligned:
            assert depth and depth.startswith("data:image/jpeg;") and rgb != depth
        else:
            assert depth is None
        assert not tools.depth.depth_models
    finally:
        tools.close()


@pytest.mark.parametrize("payload", [{}, {"depth_view": "native"}])
def test_default_observe_requires_aligned_response(payload, monkeypatch):
    client = AgentControlClient()
    request = MagicMock(return_value=payload)
    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(ValueError, match="depth_view='aligned'"):
        client.observe()
    request.assert_called_once_with("/v1/camera")


@pytest.fixture
def inference(monkeypatch):
    backend, cdm = MagicMock(), MagicMock()
    cdm.__enter__.return_value = cdm
    completed = np.full((24, 32), 0.85, np.float32)
    completed[0, 0] = np.nan
    cdm.infer.return_value = completed
    scores = np.array([0.9, 0.95, 0.85, 0.83])
    logits = np.full((4, 256, 256), -5, np.float32)
    logits[:2, 64:160, 32:120] = 5
    logits[2, 20:60, 125:180] = 5
    backend.predict.return_value = scores, logits
    monkeypatch.setattr(segmentation, "Segmenter", lambda: backend)
    factory = MagicMock(return_value=cdm)
    monkeypatch.setattr(depth_tools, "DepthCompleter", factory)
    monkeypatch.setattr(segmentation, "_GRID_COLS", 2)
    monkeypatch.setattr(segmentation, "_GRID_ROWS", 2)
    return backend, cdm, completed, logits, factory


def test_all_prompts_share_capture_and_persist_after_refresh(tmp_path, inference, frames):
    backend, cdm, completed, logits, factory = inference
    capture = PinnedObservation(MaskStore(tmp_path))
    tools = segmentation.SegmentationTools(capture)
    completion = depth_tools.DepthTools(capture)
    obs = capture.pin(frames)
    result, overlay = tools.segment(
        SegmentAtPoint(observation_id=obs, point_xy=(5, 10), return_alternatives=True)
    )
    assert [m.number for m in result.masks] == [1, 2]
    assert overlay.size == (32, 24)
    first = result.masks[0].mask_id
    snapshot = MaskStore(tmp_path).resolve(first)
    assert snapshot.completed_depth_m is None
    region = (0, 0, 15, 11)
    grid, _ = tools.segment(SegmentEverything(observation_id=obs, region_xyxy=region))
    assert len(grid.masks) == len(result.masks)
    for call in backend.predict.call_args_list[1:]:
        x, y = call.args[0][0]
        assert region[0] <= x <= region[2] and region[1] <= y <= region[3]
    box, _ = tools.segment(
        SegmentInBox(observation_id=obs, box_xyxy=(2, 2, 12, 14), return_alternatives=False)
    )
    assert len(box.masks) == 1 and backend.predict.call_args.args[1] == (2, 3)
    refined, _ = tools.segment(
        RefineMask(mask_id=first, include_points=((6, 10),), exclude_points=((25, 2),))
    )
    assert refined.masks[0].mask_id != first
    points, labels, prior = backend.predict.call_args.args
    assert points == ((5, 10), (6, 10), (25, 2)) and labels == (1, 1, 0)
    np.testing.assert_array_equal(prior, logits[1])
    factory.assert_not_called()
    info, preview = completion.complete_depth(CompleteDepth(observation_id=obs))
    assert info.observation_id == obs and preview.size == overlay.size
    assert info.valid_pixels == int(np.isfinite(completed).sum())
    assert completion.complete_depth(CompleteDepth(observation_id=obs))[0] == info
    cdm.infer.assert_called_once_with(
        frames.rgb.data, frames.registered.image.data, metres_per_unit=0.001
    )
    backend.set_image.assert_called_once()
    factory.assert_called_once_with(DepthCamera.D405)
    with pytest.raises(ValueError, match="exceeds"):
        tools.segment(RefineMask(mask_id=first, include_points=((1, 1),) * 16, exclude_points=()))
    obs2 = capture.pin(
        replace(frames, rgb=replace(frames.rgb, data=np.zeros_like(frames.rgb.data)))
    )
    assert obs != obs2
    with pytest.raises(ValueError, match="different observation"):
        tools.segment(RefineMask(mask_id=first, include_points=((2, 2),), exclude_points=()))
    with pytest.raises(ValueError, match="observation ID"):
        tools.segment(
            SegmentAtPoint(observation_id=obs, point_xy=(1, 1), return_alternatives=False)
        )
    tools.close()
    completion.close()
    saved = MaskStore(tmp_path).resolve(first)
    assert saved.observation_id == obs and saved.mask.dtype == np.bool_
    assert saved.capture.depth_frame_number == frames.depth.frame_number
    assert saved.capture.rgb_frame_number == frames.rgb.frame_number
    assert saved.capture.camera_model is DepthCamera.D405
    assert saved.capture.intrinsics == saved.intrinsics == frames.rgb.intrinsics
    np.testing.assert_array_equal(saved.mask, snapshot.mask)
    np.testing.assert_array_equal(saved.rgb, frames.rgb.data)
    np.testing.assert_array_equal(saved.completed_depth_m, completed)


def test_standalone_completion_failure_retry_and_new_capture(tmp_path, inference, frames):
    backend, cdm, depth, _, _ = inference
    capture = PinnedObservation(MaskStore(tmp_path))
    completion = depth_tools.DepthTools(capture)
    store = capture.store
    obs = capture.pin(frames)
    cdm.infer.side_effect = [RuntimeError("inference failed"), depth]
    with pytest.raises(RuntimeError, match="inference failed"):
        completion.complete_depth(CompleteDepth(observation_id=obs))
    assert store.resolve_depth(obs) is None
    info, _ = completion.complete_depth(CompleteDepth(observation_id=obs))
    backend.set_image.assert_not_called()
    assert info.observation_id == obs
    np.testing.assert_array_equal(store.resolve_depth(obs), depth)
    args = SegmentAtPoint(observation_id=obs, point_xy=(4, 8), return_alternatives=False)
    tools = segmentation.SegmentationTools(capture)
    result, _ = tools.segment(args)
    first = result.masks[0].mask_id
    np.testing.assert_array_equal(store.resolve(first).completed_depth_m, depth)
    rgb_only_view = decode_frames(encode_frames(replace(frames, registered=None)), 100)
    new = capture.pin(rgb_only_view)
    with pytest.raises(ValueError, match="observation ID"):
        completion.complete_depth(CompleteDepth(observation_id=obs))
    with pytest.raises(ValueError, match="aligned depth"):
        completion.complete_depth(CompleteDepth(observation_id=new))
    rgb_only, _ = tools.segment(
        SegmentAtPoint(observation_id=new, point_xy=args.point_xy, return_alternatives=False)
    )
    rgb_selection = store.resolve(rgb_only.masks[0].mask_id)
    assert rgb_selection.completed_depth_m is None
    assert rgb_selection.capture.camera_model is None
    assert rgb_selection.capture.depth_frame_number is None
    tools.close()
    completion.close()


def test_saved_capture_loads_original_json_format_and_rejects_invalid_metadata(tmp_path, frames):
    store = MaskStore(tmp_path)
    capture = PinnedObservation(store)
    obs = capture.pin(frames)
    capture.save()
    path = tmp_path / obs / "capture.json"
    legacy = {
        "observation_id": obs,
        "camera_model": "d405",
        "serial_number": "abc",
        "rgb_frame_id": frames.rgb.frame_id,
        "depth_frame_id": "depth",
        "rgb_frame_number": 11,
        "depth_frame_number": 12,
        "depth_scale_m_per_unit": 0.001,
        "rgb_device_timestamp": 1,
        "depth_device_timestamp": 1,
        "rgb_timestamp_domain": "device",
        "depth_timestamp_domain": "device",
        "depth_units": "metres",
        "intrinsics": {
            "width": 32,
            "height": 24,
            "fx": 32,
            "fy": 33,
            "cx": 15,
            "cy": 11,
            "distortion_model": "none",
            "distortion_coefficients": [],
        },
    }
    assert json.loads(path.read_text()) == legacy
    path.write_text(json.dumps(legacy))
    metres = np.ones((24, 32), np.float32)
    store.save_depth(obs, metres)
    np.testing.assert_array_equal(store.resolve_depth(obs), metres)
    path.write_text(json.dumps({**legacy, "rgb_frame_number": "11"}))
    with pytest.raises(ValueError, match="rgb_frame_number"):
        store.save_depth(obs, metres)
