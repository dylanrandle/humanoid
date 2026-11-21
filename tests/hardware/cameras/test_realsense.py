"""SDK-independent acquisition and camera-node behavior."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from humanoid.config.camera import REALSENSE_CAMERA_CONFIG
from humanoid.constants import Topic
from humanoid.hardware.cameras.realsense import RealSenseCamera
from humanoid.middleware.publisher import Publisher
from humanoid.nodes.robot.camera import CameraDriverNode
from humanoid.types.camera import CameraEncoding


def make_camera():
    config = replace(REALSENSE_CAMERA_CONFIG, width=4, height=2, serial_number="test-camera")
    sdk = MagicMock()
    pipeline = sdk.pipeline.return_value
    profile = pipeline.start.return_value
    device = profile.get_device.return_value
    device.first_depth_sensor.return_value.get_depth_scale.return_value = 0.001
    device.get_info.return_value = "test-camera"
    video = profile.get_stream.return_value.as_video_stream_profile.return_value
    video.get_intrinsics.return_value = SimpleNamespace(
        width=4, height=2, fx=3, fy=3, ppx=1.5, ppy=0.5, model="none", coeffs=[0] * 5
    )
    frames = pipeline.poll_for_frames.return_value
    for frame, pixels in (
        (frames.get_color_frame.return_value, np.arange(24, dtype=np.uint8).reshape(2, 4, 3)),
        (frames.get_depth_frame.return_value, np.array([[0, 1, 1000, 65535]] * 2, dtype=np.uint16)),
    ):
        frame.get_data.return_value = pixels
        frame.get_timestamp.return_value = 1234.5
        frame.get_frame_timestamp_domain.return_value = "hardware_clock"
        frame.get_frame_number.return_value = 7
    clock = MagicMock(return_value=5.0)
    return RealSenseCamera(config, sdk=sdk, clock=clock), sdk, clock


def test_native_streams_metadata_and_owned_pixels():
    camera, sdk, _ = make_camera()
    camera.start()
    frames = camera.read()
    assert frames is not None
    settings = sdk.config.return_value
    settings.enable_device.assert_called_once_with("test-camera")
    assert settings.enable_stream.call_args_list[0].args == (
        sdk.stream.color,
        4,
        2,
        sdk.format.rgb8,
        30,
    )
    assert settings.enable_stream.call_args_list[1].args == (
        sdk.stream.depth,
        4,
        2,
        sdk.format.z16,
        30,
    )
    sdk.align.assert_not_called()
    assert frames.rgb.encoding is CameraEncoding.RGB8
    assert frames.depth.encoding is CameraEncoding.DEPTH16
    assert frames.rgb.frame_id == camera.config.rgb_frame
    assert frames.depth.frame_id == camera.config.depth_frame
    assert frames.depth.depth_scale == pytest.approx(0.001)
    assert frames.rgb.timestamp == frames.depth.timestamp
    assert frames.rgb.device_timestamp == pytest.approx(1.2345)
    assert frames.depth.timestamp_domain == "hardware_clock"
    assert frames.depth.intrinsics.fx == pytest.approx(3)
    sdk_frames = sdk.pipeline.return_value.poll_for_frames.return_value
    pixels = sdk_frames.get_depth_frame.return_value.get_data.return_value
    original = pixels.copy()
    pixels[:] = 0
    np.testing.assert_array_equal(frames.depth.data, original)
    camera.stop()
    camera.stop()
    sdk.pipeline.return_value.stop.assert_called_once()


def test_missing_frames_timeout_and_startup_cleanup():
    camera, sdk, clock = make_camera()
    with pytest.raises(RuntimeError, match="not streaming"):
        camera.read()
    camera.start()
    sdk.pipeline.return_value.poll_for_frames.return_value = None
    assert camera.read() is None
    clock.return_value = 9.0
    with pytest.raises(RuntimeError, match="timed out"):
        camera.read()
    camera.stop()
    camera, sdk, _ = make_camera()
    sdk.pipeline.return_value.start.return_value.get_device.side_effect = RuntimeError("disconnect")
    with pytest.raises(RuntimeError, match="disconnect"):
        camera.start()
    sdk.pipeline.return_value.stop.assert_called_once()


def test_driver_publishes_complete_pairs_and_closes():
    camera, sdk, _ = make_camera()
    publisher = MagicMock(spec=Publisher)
    node = CameraDriverNode(camera.config, camera=camera, publisher=publisher)
    node.setup()
    node.step()
    assert [c.kwargs["topic"] for c in publisher.publish.call_args_list] == [
        Topic.CAMERA_RGB,
        Topic.CAMERA_DEPTH,
    ]
    publisher.reset_mock()
    sdk.pipeline.return_value.poll_for_frames.return_value = None
    node.step()
    publisher.publish.assert_not_called()
    node.on_close()
    sdk.pipeline.return_value.stop.assert_called_once()


def test_real_camera_fails_clearly_on_macos_without_importing_sdk(monkeypatch):
    monkeypatch.setattr("humanoid.hardware.cameras.realsense.sys.platform", "darwin")
    importer = MagicMock()
    monkeypatch.setattr("humanoid.hardware.cameras.realsense.importlib.import_module", importer)
    camera = RealSenseCamera(REALSENSE_CAMERA_CONFIG)
    with pytest.raises(RuntimeError, match="not supported on macOS"):
        camera.start()
    importer.assert_not_called()


@pytest.mark.parametrize("model", ["D405", "D435"])
def test_known_model_aligns_the_same_sdk_frameset(model):
    camera, sdk, _ = make_camera()
    profile = sdk.pipeline.return_value.start.return_value
    profile.get_device.return_value.get_info.side_effect = lambda key: (
        f"Intel RealSense {model}" if key == sdk.camera_info.name else "test-camera"
    )
    units = 1500
    aligned = np.full((2, 4), units, dtype=np.uint16)
    registered = sdk.align.return_value.process.return_value.get_depth_frame.return_value
    registered.get_data.return_value = aligned
    camera.start()
    frames = camera.read()
    sdk.align.assert_called_once_with(sdk.stream.color)
    sdk.align.return_value.process.assert_called_once_with(
        sdk.pipeline.return_value.poll_for_frames.return_value
    )
    assert frames.registered.camera_model == model.lower()
    aligned[:] = 0
    assert np.all(frames.registered.image.data == units)
    assert not np.array_equal(frames.depth.data, frames.registered.image.data)
    camera.stop()
