import time

import numpy as np

from humanoid.remote_control.serialization import (
    decode_frames,
    decode_state,
    encode_frames,
    encode_state,
)
from humanoid.types.camera import CameraEncoding, CameraFrames, CameraImage, CameraIntrinsics


def test_state_uses_client_clock_and_preserves_unavailable_efforts(service):
    state = service.observation().robot_state
    state.joint_efforts = np.full_like(state.joint_velocities, np.nan)
    payload = encode_state(state)
    assert all(value is None for value in payload["joint_efforts"])
    client_time = 100.0
    restored = decode_state(payload, received_after=client_time)
    assert restored.timestamp <= client_time
    np.testing.assert_allclose(restored.joint_positions, state.joint_positions)
    assert restored.joint_efforts is not None
    assert np.isnan(restored.joint_efforts).all()


def test_native_rgb_depth_round_trip_preserves_pixels_and_calibration():
    intrinsics = CameraIntrinsics(
        width=2,
        height=2,
        fx=100,
        fy=100,
        cx=1,
        cy=1,
        distortion_model="none",
        distortion_coefficients=(),
    )
    now = time.perf_counter()
    frames = CameraFrames(
        rgb=CameraImage(
            timestamp=now,
            device_timestamp=12.0,
            timestamp_domain="hardware_clock",
            frame_number=7,
            serial_number="test",
            intrinsics=intrinsics,
            frame_id="rgb",
            encoding=CameraEncoding.RGB8,
            data=np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        ),
        depth=CameraImage(
            timestamp=now,
            device_timestamp=12.0,
            timestamp_domain="hardware_clock",
            frame_number=7,
            serial_number="test",
            intrinsics=intrinsics,
            frame_id="depth",
            encoding=CameraEncoding.DEPTH16,
            data=np.array([[0, 1000], [65535, 2500]], dtype=np.uint16),
            depth_scale=0.001,
        ),
    )
    client_time = 100.0
    restored = decode_frames(encode_frames(frames), received_after=client_time)
    for actual, expected in ((restored.rgb, frames.rgb), (restored.depth, frames.depth)):
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual.intrinsics == expected.intrinsics
        assert actual.frame_id == expected.frame_id
        assert actual.depth_scale == expected.depth_scale
        assert actual.timestamp <= client_time
