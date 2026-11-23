"""Record the simulated room and the detail camera side by side."""

import math
import subprocess
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import cv2
import mujoco
import numpy as np

from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.render_stream import SimulationRenderStream
from humanoid.types.simulation import RenderFrame


class EpisodeVideo:
    fps = 12
    width = 480
    height = 360

    def __init__(
        self,
        engine: NativeMujocoEngine,
        path: Path,
        *,
        overview: mujoco.MjvCamera,
        follow_body: str | None = None,
        start_time: float | None = None,
    ):
        self.engine = engine
        self.title = path.stem.replace("-", " / seed ")
        self.renderer = mujoco.Renderer(engine.model, height=self.height, width=self.width)
        self.overview = overview
        self.views: list[str | mujoco.MjvCamera] = [self.overview]
        self.labels = []
        for name, camera in engine.cameras.items():
            self.views.append(camera.config.rgb_frame)
            self.labels.append(f"{name.capitalize()} {camera.config.model.upper()}")
        if follow_body is not None:
            self.detail = mujoco.MjvCamera()
            self.detail.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.detail.trackbodyid = engine.model.body(follow_body).id
            self.detail.lookat[:] = engine.data.body(follow_body).xpos
            self.detail.distance = 0.36
            self.detail.azimuth = 62
            self.detail.elevation = -26
            self.views = [self.overview, self.detail]
            self.labels = [f"{follow_body} close-up"]
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.start_time = engine.data.time if start_time is None else start_time
        self._written_frames = 0
        self._last_frame = b""
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.width * len(self.views)}x{self.height + 32}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-n",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def capture(self, action: str, data: mujoco.MjData | None = None) -> None:
        if data is None:
            data = self.engine.data
        t = data.time
        frame_index = math.floor((t - self.start_time) * self.fps + 1e-9)
        if frame_index < self._written_frames:
            return
        views = []
        for camera in self.views:
            self.renderer.update_scene(data, camera=camera, scene_option=self.option)
            views.append(self.renderer.render().copy())
        header = np.full((32, len(self.views) * self.width, 3), 28, dtype=np.uint8)
        captions = [f"{self.title}    {t:.1f}s    {action}", *self.labels]
        for index, caption in enumerate(captions):
            x = index * self.width + 8
            cv2.putText(
                header,
                caption,
                (x, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        frame = np.vstack((header, np.hstack(views)))
        assert self.process.stdin is not None
        pixels = frame.tobytes()
        # Preserve playback duration when a busy renderer skips a snapshot.
        for _ in range(self._written_frames, frame_index):
            self.process.stdin.write(self._last_frame or pixels)
        self.process.stdin.write(pixels)
        self._last_frame = pixels
        self._written_frames = frame_index + 1

    def close(self) -> None:
        try:
            assert self.process.stdin is not None
            self.process.stdin.close()
            if self.process.wait():
                raise RuntimeError("ffmpeg failed to encode the episode.")
        finally:
            self.renderer.close()


def episode_video_stream(
    engine: NativeMujocoEngine, path: Path, overview: mujoco.MjvCamera
) -> SimulationRenderStream:
    start = engine.data.time

    @contextmanager
    def recorder() -> Iterator[RenderFrame]:
        with closing(EpisodeVideo(engine, path, overview=overview, start_time=start)) as video:
            yield lambda data, _: video.capture("simulation", data)

    return SimulationRenderStream("video", engine.model, EpisodeVideo.fps, recorder())
