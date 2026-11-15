"""Lossless camera wire format and shared publisher/subscriber contracts."""

import zlib
from unittest.mock import MagicMock

import numpy as np
import pytest

from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.camera import CameraEncoding, CameraImage, CameraIntrinsics
from humanoid.types.lcm import camera_image_t
from humanoid.types.lcm.converter import LCMConverter


def make_image(encoding):
    depth = encoding is CameraEncoding.DEPTH16
    return CameraImage(
        timestamp=1.25,
        device_timestamp=0.125,
        timestamp_domain="hardware_clock",
        frame_number=17,
        frame_id="depth" if depth else "rgb",
        serial_number="D435",
        encoding=encoding,
        data=(
            np.array([[0, 1, 1000, 65535]], dtype=np.uint16)
            if depth
            else np.arange(12, dtype=np.uint8).reshape(1, 4, 3)
        ),
        intrinsics=CameraIntrinsics(
            width=4,
            height=1,
            fx=3,
            fy=3,
            cx=1.5,
            cy=0,
            distortion_model="none",
            distortion_coefficients=(0.0,) * 5,
        ),
        depth_scale=0.001 if depth else 0,
    )


@pytest.mark.parametrize("encoding", list(CameraEncoding))
def test_wire_roundtrip_and_subscriber(encoding, monkeypatch):
    original = make_image(encoding)
    topic = Topic.CAMERA_RGB if encoding is CameraEncoding.RGB8 else Topic.CAMERA_DEPTH
    transport = MagicMock()
    monkeypatch.setattr("humanoid.middleware.publisher.lcm.LCM", lambda url: transport)
    # Avoid a transport thread; exercise actual receive queues and decoding below.
    monkeypatch.setattr("humanoid.middleware.subscriber.threading.Thread.start", lambda self: None)
    monkeypatch.setattr(
        "humanoid.middleware.subscriber.threading.Thread.join", lambda self, timeout=None: None
    )
    publisher = Publisher()
    subscriber = Subscriber([topic])
    try:
        publisher.publish(original, topic)
        channel, payload = transport.publish.call_args.args
        assert camera_image_t.decode(payload).compression == "lz4"
        subscriber._handle_message(channel, payload)
        recovered = subscriber.receive(topic)
        assert isinstance(recovered, CameraImage)
        np.testing.assert_array_equal(recovered.data, original.data)
        assert recovered.intrinsics == original.intrinsics
        for field in (
            "timestamp",
            "device_timestamp",
            "timestamp_domain",
            "frame_number",
            "frame_id",
            "serial_number",
            "encoding",
            "depth_scale",
        ):
            assert getattr(recovered, field) == getattr(original, field)
        wrong_topic = Topic.CAMERA_DEPTH if topic is Topic.CAMERA_RGB else Topic.CAMERA_RGB
        with pytest.raises(ValueError, match="requires"):
            publisher.publish(original, wrong_topic)
    finally:
        subscriber.close()


@pytest.mark.parametrize("codec", ["lz4", "zlib"])
@pytest.mark.parametrize(
    "mutation", ["truncated", "dimensions", "compression", "trailing", "oversized"]
)
def test_rejects_invalid_camera_payload(mutation, codec):
    message = LCMConverter.camera_image_to_lcm(make_image(CameraEncoding.DEPTH16))
    if codec == "zlib":
        message.compression = "zlib"
        message.data = zlib.compress(make_image(CameraEncoding.DEPTH16).data.tobytes())
        message.data_size = len(message.data)
    message = camera_image_t.decode(message.encode())
    if mutation == "truncated":
        message.data = message.data[:-1]
    elif mutation == "dimensions":
        message.width += 1
    elif mutation == "compression":
        message.compression = "jpeg"
    elif mutation == "oversized":
        message.width -= 1
    else:
        message.data += b"extra"
    with pytest.raises(ValueError):
        LCMConverter.camera_image_from_lcm(message)


@pytest.mark.parametrize("encoding", list(CameraEncoding))
def test_legacy_zlib_recordings_remain_readable(encoding):
    original = make_image(encoding)
    message = LCMConverter.camera_image_to_lcm(original)
    message.compression = "zlib"
    dtype = "u1" if encoding is CameraEncoding.RGB8 else "<u2"
    message.data = zlib.compress(original.data.astype(dtype, copy=False).tobytes())
    message.data_size = len(message.data)
    recovered = LCMConverter.camera_image_from_lcm(camera_image_t.decode(message.encode()))
    np.testing.assert_array_equal(recovered.data, original.data)
