"""Export RGB and colorized depth streams from an LCM recording to H.264 MP4s."""

import argparse
import struct
import subprocess
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

import lcm
import numpy as np

from humanoid.config.camera import CAMERA_DISPLAY_CONFIG
from humanoid.perception.depth import colorize_depth
from humanoid.recording import RECORDING_LOG_FILENAME
from humanoid.types.camera import (
    CameraEncoding,
    CameraImage,
    CameraTopic,
    RegisteredDepth,
    camera_topics,
)
from humanoid.types.lcm import camera_image_t, registered_depth_t
from humanoid.types.lcm.converter import LCMConverter

_US_PER_SECOND = 1_000_000


def _video_channel(
    channel: str,
) -> tuple[str, CameraTopic[CameraImage] | CameraTopic[RegisteredDepth]] | None:
    try:
        _, camera, stream = channel.split("/")
        topics = camera_topics(camera)
    except ValueError:
        return None
    for topic in (topics.rgb, topics.depth, topics.registered_depth):
        if channel == topic.value:
            return f"{camera.lower()}.{stream.lower()}.mp4", topic
    return None


def _pixels(
    topic: CameraTopic[CameraImage] | CameraTopic[RegisteredDepth], data: bytes
) -> np.ndarray:
    if topic.message_type is RegisteredDepth:
        message = registered_depth_t.decode(data).image
        expected = CameraEncoding.DEPTH16
    else:
        message = camera_image_t.decode(data)
        expected = topic.encoding
    image = LCMConverter.camera_image_from_lcm(message)
    if image.encoding != expected:
        raise ValueError(f"Expected {expected} pixels, got {image.encoding}")
    if image.encoding == CameraEncoding.RGB8:
        return image.data
    return colorize_depth(
        image.data.astype(np.float32) * image.depth_scale,
        CAMERA_DISPLAY_CONFIG.depth_near_meters,
        CAMERA_DISPLAY_CONFIG.default_depth_far_meters,
    )


def _frames(log: lcm.EventLog) -> Iterator[tuple[str, int, np.ndarray]]:
    for event in iter(log.read_next_event, None):
        video_channel = _video_channel(event.channel)
        if video_channel is None:
            continue
        filename, topic = video_channel
        try:
            pixels = _pixels(topic, event.data)
        except (ValueError, RuntimeError, struct.error) as exc:
            raise ValueError(f"{event.channel} at event {event.eventnum}: {exc}") from exc
        yield filename, event.timestamp, pixels


@contextmanager
def _encoder(path: Path, width: int, height: int, fps: int) -> Iterator[BinaryIO]:
    args = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as process:
        assert process.stdin is not None
        with process.stdin as pipe:
            yield pipe
        if process.wait():
            raise RuntimeError(f"FFmpeg failed to encode {path.name}")


@dataclass
class _Video:
    pipe: BinaryIO
    frame: np.ndarray
    frames_written: int = 0

    def write_until(self, index: int) -> None:
        while self.frames_written < index:
            self.pipe.write(memoryview(self.frame).cast("B"))
            self.frames_written += 1


def export(recording: Path, output: Path | None = None, fps: int = 30) -> list[Path]:
    if fps <= 0:
        raise ValueError("Playback frame rate must be positive")
    if recording.is_dir():
        recording = recording / RECORDING_LOG_FILENAME
    if not recording.is_file():
        raise FileNotFoundError(f"LCM recording not found: {recording}")

    output = output or recording.with_name(f"{recording.stem}-videos")
    output.mkdir(parents=True, exist_ok=True)
    videos: dict[Path, _Video] = {}
    start_us: int | None = None
    last_index = 0
    with TemporaryDirectory(dir=output) as temporary:
        with closing(lcm.EventLog(recording, "r")) as log, ExitStack() as encoders:
            for filename, timestamp, pixels in _frames(log):
                if start_us is None:
                    start_us = timestamp
                last_index = max(last_index, round((timestamp - start_us) * fps / _US_PER_SECOND))
                path = output / filename
                if path not in videos:
                    if path.exists():
                        raise FileExistsError(f"Video already exists: {path}")
                    height, width = pixels.shape[:2]
                    pipe = encoders.enter_context(
                        _encoder(Path(temporary) / path.name, width, height, fps)
                    )
                    videos[path] = _Video(pipe, np.zeros_like(pixels))
                video = videos[path]
                if pixels.shape != video.frame.shape:
                    raise ValueError(f"Resolution changed while exporting {filename}")
                video.write_until(last_index)
                video.frame = pixels
            if not videos:
                raise ValueError(f"No camera frames found in {recording}")
            for video in videos.values():
                video.write_until(last_index + 1)
        for path in videos:
            path.hardlink_to(Path(temporary) / path.name)
    return sorted(videos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="LCM log file or recording bundle directory")
    parser.add_argument("--output", type=Path, help="Directory for the camera videos")
    parser.add_argument(
        "--fps", type=int, default=30, help="Output frames per second (default: 30)"
    )
    args = parser.parse_args()
    try:
        paths = export(args.recording, args.output, args.fps)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
