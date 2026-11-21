import time
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest

from humanoid.agent.client import AgentControlClient
from humanoid.agent.serialization import decode_frames, encode_frames
from humanoid.agent.server import AgentControlServer
from humanoid.config.camera import REALSENSE_CAMERA_CONFIG
from humanoid.types.agent import AgentControlEndpoint
from humanoid.types.camera import (
    CameraEncoding,
    CameraFrames,
    CameraImage,
    CameraIntrinsics,
    RegisteredDepth,
)
from humanoid.types.depth import DepthCamera


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


@pytest.mark.parametrize("payload", [{}, {"depth_view": "native"}])
def test_default_observe_requires_aligned_response(payload, monkeypatch):
    client = AgentControlClient()
    request = MagicMock(return_value=payload)
    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(ValueError, match="depth_view='aligned'"):
        client.observe()
    request.assert_called_once_with("/v1/camera")
