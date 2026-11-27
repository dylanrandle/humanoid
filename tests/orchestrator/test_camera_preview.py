"""Camera preview freshness, color scale, and source data preservation."""

from unittest.mock import MagicMock

import numpy as np

from humanoid.api.services.cameras import CameraFrameService
from humanoid.middleware.subscriber import Subscriber
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


def test_sources_share_latest_frames_and_expire_by_local_receipt():
    now = 100.0
    subscriber = MagicMock(spec=Subscriber)
    rgb, depth = _frame(CameraEncoding.RGB8), _frame(CameraEncoding.DEPTH16)
    subscriber.receive_with_timestamp.side_effect = [(rgb, now), (depth, now), None, None]
    source = CameraFrameService(camera_name="base", subscriber=subscriber, clock=lambda: now)
    assert source.read_frame("rgb") is rgb
    assert source.read_frame("depth") is depth
    assert source.read_frame("rgb") is rgb
    now += 3
    assert source.read_frame("depth") is None
    source.close()
    subscriber.close.assert_called_once()


def test_invalid_encoding_is_not_exposed_to_video_encoder():
    subscriber = MagicMock(spec=Subscriber)
    subscriber.receive_with_timestamp.return_value = (_frame(CameraEncoding.DEPTH16), 1)
    source = CameraFrameService(camera_name="base", subscriber=subscriber, clock=lambda: 1)
    assert source.read_frame("rgb") is None
    subscriber.receive_with_timestamp.assert_called_once_with(camera_topics("base").rgb)
