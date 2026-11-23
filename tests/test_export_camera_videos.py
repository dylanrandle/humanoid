import shutil
import subprocess
from contextlib import closing
from pathlib import Path

import lcm
import numpy as np
import pytest

from humanoid.types.camera import CameraEncoding, CameraImage, CameraIntrinsics
from humanoid.types.lcm import registered_depth_t
from humanoid.types.lcm.converter import LCMConverter
from scripts.export_camera_videos import export


def _message(encoding: CameraEncoding):
    rgb = encoding == CameraEncoding.RGB8
    return LCMConverter.camera_image_to_lcm(
        CameraImage(
            timestamp=0,
            device_timestamp=0,
            timestamp_domain="simulation_clock",
            frame_number=1,
            frame_id="test",
            serial_number="test",
            encoding=encoding,
            data=np.full((15, 17, 3), [255, 0, 0], np.uint8)
            if rgb
            else np.full((15, 17), 500, np.uint16),
            intrinsics=CameraIntrinsics(
                width=17,
                height=15,
                fx=10,
                fy=10,
                cx=8,
                cy=7,
                distortion_model="none",
                distortion_coefficients=(),
            ),
            depth_scale=0 if rgb else 0.001,
        )
    )


def _frames(path: Path) -> np.ndarray:
    pixels = subprocess.check_output(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    return np.frombuffer(pixels, np.uint8).reshape(-1, 16, 18, 3)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="Video export requires FFmpeg")
def test_exports_synchronized_rgb_native_and_registered_depth(tmp_path):
    rgb = _message(CameraEncoding.RGB8).encode()
    depth = _message(CameraEncoding.DEPTH16)
    registered = registered_depth_t()
    registered.camera_model = "D435"
    registered.image = depth
    with closing(lcm.EventLog(tmp_path / "recording.lcm", "w")) as log:
        log.write_event(0, "ROBOT_STATE", b"ignored")
        log.write_event(100_000, "CAMERA/BASE/RGB", rgb)
        log.write_event(300_000, "CAMERA/WRIST/DEPTH", depth.encode())
        log.write_event(300_000, "CAMERA/WRIST/REGISTERED_DEPTH", registered.encode())
        log.write_event(400_000, "CAMERA/BASE/RGB", rgb)

    paths = export(tmp_path, fps=10)
    assert [p.name for p in paths] == [
        "base.rgb.mp4",
        "wrist.depth.mp4",
        "wrist.registered_depth.mp4",
    ]
    color, native, aligned = map(_frames, paths)
    expected_frames = 4  # 0.0, 0.1, 0.2, 0.3 s
    assert len(color) == len(native) == len(aligned) == expected_frames
    for frame in color:
        np.testing.assert_allclose(frame[5, 5], [255, 0, 0], atol=8)
    for frames in (native, aligned):
        for frame in frames[:2]:
            assert not frame.any()
        np.testing.assert_allclose(frames[2:, 5, 5], [[205, 193, 92]] * 2, atol=10)

    with pytest.raises(FileExistsError, match="already exists"):
        export(tmp_path, fps=10)


def test_reports_logs_without_camera_frames(tmp_path):
    log_path = tmp_path / "robot.lcm"
    with closing(lcm.EventLog(log_path, "w")) as log:
        log.write_event(0, "ROBOT_STATE", b"ignored")
    with pytest.raises(ValueError, match="No camera frames"):
        export(log_path)
