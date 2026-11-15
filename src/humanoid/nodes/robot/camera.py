"""Managed RealSense RGB and native depth publisher."""

from humanoid.config.camera import REALSENSE_CAMERA_CONFIG
from humanoid.constants import Topic
from humanoid.hardware.cameras.realsense import RealSenseCamera
from humanoid.middleware.publisher import Publisher
from humanoid.nodes.base import Node
from humanoid.types.camera import CameraSource, RealSenseCameraConfig


class CameraDriverNode(Node):
    def __init__(
        self,
        config: RealSenseCameraConfig = REALSENSE_CAMERA_CONFIG,
        *,
        camera: CameraSource | None = None,
        publisher: Publisher | None = None,
    ) -> None:
        self.rate_hz = float(config.fps)
        self.camera = camera if camera is not None else RealSenseCamera(config)
        self.publisher = publisher if publisher is not None else Publisher()

    def setup(self) -> None:
        self.camera.start()

    def step(self) -> None:
        frames = self.camera.read()
        if frames is None:
            return
        self.publisher.publish(frames.rgb, topic=Topic.CAMERA_RGB)
        self.publisher.publish(frames.depth, topic=Topic.CAMERA_DEPTH)

    def on_close(self) -> None:
        self.camera.stop()
