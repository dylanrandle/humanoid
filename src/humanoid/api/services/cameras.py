"""Shared latest-frame camera sources for WebRTC viewers."""

import threading
import time
from collections.abc import Callable
from typing import Literal

from humanoid.config.camera import CAMERA_DISPLAY_CONFIG
from humanoid.constants import DEFAULT_LCM_URL
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.camera import (
    CameraDisplayConfig,
    CameraImage,
    CameraTopic,
    camera_topics,
)


class CameraFrameService:
    def __init__(
        self,
        subscriber: Subscriber | None = None,
        config: CameraDisplayConfig = CAMERA_DISPLAY_CONFIG,
        clock: Callable[[], float] = time.perf_counter,
        *,
        camera_name: str,
        lcm_url: str = DEFAULT_LCM_URL,
    ) -> None:
        self.topics = camera_topics(camera_name)
        self._subscriber = subscriber or Subscriber(
            topics=[self.topics.rgb, self.topics.depth], url=lcm_url
        )
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._frames: dict[CameraTopic[CameraImage], tuple[CameraImage, float]] = {}

    def read_frame(self, stream: Literal["rgb", "depth"]) -> CameraImage | None:
        """Read the most recent raw frame without JPEG encoding or buffering history."""
        topic = self.topics.rgb if stream == "rgb" else self.topics.depth
        with self._lock:
            received = self._subscriber.receive_with_timestamp(topic)
            if received is not None:
                frame, received_at = received
                if isinstance(frame, CameraImage) and frame.encoding is topic.encoding:
                    self._frames[topic] = (frame, received_at)
            latest = self._frames.get(topic)
            if latest is None or self._clock() - latest[1] > self._config.max_age_seconds:
                return None
            return latest[0]

    def close(self) -> None:
        self._subscriber.close()
        with self._lock:
            self._frames.clear()
