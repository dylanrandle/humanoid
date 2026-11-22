"""Publish each configured hardware camera on its named LCM channels."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

from humanoid.config.selection import ROBOT_CONFIG
from humanoid.hardware.cameras.factory import create_camera
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.nodes.base import Node
from humanoid.types.camera import CameraSource, HardwareCameraConfig, camera_topics
from humanoid.types.robot import RobotConfig
from humanoid.utils.loop import loop_at_rate

logger = get_logger(__name__)


class CameraDriverNode(Node):
    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        *,
        camera: CameraSource | None = None,
        publisher: Publisher | None = None,
    ) -> None:
        devices = [
            (name, mount.hardware)
            for name, mount in robot_config.cameras.items()
            if mount.hardware is not None
        ]
        if not devices:
            raise ValueError(f"{robot_config.name} has no configured hardware cameras.")
        (name, config), *self.additional = devices
        self.topics = camera_topics(name)
        self.rate_hz = float(config.fps)
        self.camera = camera if camera is not None else create_camera(config)
        self.publisher = publisher if publisher is not None else Publisher()
        self._stop = Event()
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.additional)))

    def _stream(self, name: str, config: HardwareCameraConfig) -> None:
        # Each additional camera gets independent startup, polling and transport.
        camera = create_camera(config)
        topics = camera_topics(name)
        publisher = Publisher(self.publisher.url)

        def publish() -> None:
            frames = camera.read()
            if frames is not None:
                publisher.publish_camera(frames, topics)

        while not self._stop.is_set():
            try:
                camera.start()
                loop_at_rate(publish, config.fps, stop_condition=self._stop.is_set)
            except RuntimeError as exc:
                logger.warning("%s camera unavailable: %s", name, exc)
            finally:
                camera.stop()
            self._stop.wait(5)

    def setup(self) -> None:
        self.camera.start()
        self._streams = [
            self._pool.submit(self._stream, name, config) for name, config in self.additional
        ]

    def step(self) -> None:
        for stream in self._streams:
            if stream.done():
                stream.result()
        frames = self.camera.read()
        if frames is not None:
            self.publisher.publish_camera(frames, self.topics)

    def on_close(self) -> None:
        self._stop.set()
        try:
            self.camera.stop()
        finally:
            self._pool.shutdown(wait=True)
