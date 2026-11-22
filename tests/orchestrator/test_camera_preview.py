"""Camera preview freshness, color scale, and source data preservation."""

import base64
import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from humanoid.middleware.subscriber import Subscriber
from humanoid.orchestrator.monitor.camera import CameraPreviewMonitor
from humanoid.types.camera import CameraEncoding, CameraImage, CameraIntrinsics, camera_topics


def _frame(encoding):
    pixels = (
        np.full((2, 4, 3), [255, 0, 0], dtype=np.uint8)
        if encoding is CameraEncoding.RGB8
        else np.array([[0, 200, 1000, 3000]] * 2, dtype=np.uint16)
    )
    return CameraImage(
        timestamp=-1000,
        device_timestamp=10,
        timestamp_domain="device",
        frame_number=2,
        frame_id="optical_frame",
        serial_number="camera",
        encoding=encoding,
        data=pixels,
        intrinsics=CameraIntrinsics(
            width=4,
            height=2,
            fx=3,
            fy=3,
            cx=1.5,
            cy=0.5,
            distortion_model="none",
            distortion_coefficients=(),
        ),
        depth_scale=0.001 if encoding is CameraEncoding.DEPTH16 else 0,
    )


def test_preview_encodes_both_streams_and_expires_by_local_receipt():
    now = 100.0
    subscriber = MagicMock(spec=Subscriber)
    rgb = _frame(CameraEncoding.RGB8)
    depth = _frame(CameraEncoding.DEPTH16)
    original_depth = depth.data.copy()
    subscriber.receive_with_timestamp.side_effect = [
        (rgb, now),
        (depth, now),
        None,
        None,
        None,
        None,
    ]
    monitor = CameraPreviewMonitor(camera_name="base", subscriber=subscriber, clock=lambda: now)
    snapshot = monitor.snapshot(3.0)
    assert snapshot.rgb.connected and snapshot.depth.connected
    assert snapshot.rgb.age_seconds == 0
    assert snapshot.depth.frame_id == "optical_frame"
    assert snapshot.rgb.image is not None
    image = Image.open(io.BytesIO(base64.b64decode(snapshot.rgb.image.split(",")[1])))
    assert image.size == (4, 2)
    assert np.asarray(image)[0, 0, 0] > np.asarray(image)[0, 0, 1]
    np.testing.assert_array_equal(depth.data, original_depth)
    now += 1
    cached = monitor.snapshot(3.0)
    assert cached.rgb.image == snapshot.rgb.image
    now += 2
    stale = monitor.snapshot(3.0)
    assert not stale.rgb.connected and stale.rgb.image is None
    assert not stale.depth.connected and stale.depth.image is None
    assert stale.depth.age_seconds == pytest.approx(3.0)
    monitor.close()
    subscriber.close.assert_called_once()


def test_depth_colors_use_scale_and_range_without_modifying_samples():
    subscriber = MagicMock(spec=Subscriber)
    frame = _frame(CameraEncoding.DEPTH16)
    subscriber.receive_with_timestamp.side_effect = [None, (frame, 1), None, None]
    monitor = CameraPreviewMonitor(camera_name="base", subscriber=subscriber, clock=lambda: 1)
    first = monitor.snapshot(3)
    colors = monitor._colorize_depth(frame)
    np.testing.assert_array_equal(colors[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(colors[0, 1], [248, 196, 66])
    np.testing.assert_array_equal(colors[0, 3], [51, 71, 169])
    second = monitor.snapshot(10)
    assert second.depth.image != first.depth.image
    assert not first.rgb.connected
    assert second.depth_far_meters == pytest.approx(10)


def test_reset_and_wrong_encoding_never_show_previous_frames():
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive_with_timestamp.side_effect = [
        (_frame(CameraEncoding.DEPTH16), 1),
        None,  # Wrong topic.
        None,
        (_frame(CameraEncoding.DEPTH16), 1),  # Reset drains a pending frame.
        None,
        None,
    ]
    monitor = CameraPreviewMonitor(camera_name="base", subscriber=subscriber, clock=lambda: 1)
    assert not monitor.snapshot(3).rgb.connected
    monitor.reset()
    snapshot = monitor.snapshot(3)
    assert not snapshot.rgb.connected and not snapshot.depth.connected
    subscriber.receive_with_timestamp.assert_any_call(camera_topics("base").depth)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 2])
def test_rejects_invalid_depth_range(value):
    monitor = CameraPreviewMonitor(camera_name="base", subscriber=MagicMock(spec=Subscriber))
    with pytest.raises(ValueError, match="range"):
        monitor.snapshot(value)
