"""Latest-frame camera previews for the browser, separate from control telemetry."""

import base64
import io
import threading
import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from PIL import Image

from humanoid.config.camera import CAMERA_PREVIEW_CONFIG
from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.perception.depth import colorize_depth
from humanoid.types.camera import (
    CameraEncoding,
    CameraImage,
    CameraPreview,
    CameraPreviewConfig,
    CameraPreviewSnapshot,
)


class CameraPreviewMonitor:
    def __init__(
        self,
        subscriber: Subscriber | None = None,
        config: CameraPreviewConfig = CAMERA_PREVIEW_CONFIG,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._subscriber = subscriber or Subscriber(topics=[Topic.CAMERA_RGB, Topic.CAMERA_DEPTH])
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._frames: dict[Topic, tuple[CameraImage, float]] = {}
        self._previews: dict[Topic, CameraPreview] = {}
        self._depth_far = config.default_depth_far_meters

    def snapshot(self, depth_far: float) -> CameraPreviewSnapshot:
        if depth_far not in self._config.depth_ranges_meters:
            raise ValueError("Unsupported depth preview range.")
        with self._lock:
            if depth_far != self._depth_far:
                self._previews.pop(Topic.CAMERA_DEPTH, None)
                self._depth_far = depth_far
            previews = {}
            for topic, encoding in (
                (Topic.CAMERA_RGB, CameraEncoding.RGB8),
                (Topic.CAMERA_DEPTH, CameraEncoding.DEPTH16),
            ):
                received = self._subscriber.receive_with_timestamp(topic)
                if received is not None:
                    frame, received_at = received
                    if isinstance(frame, CameraImage) and frame.encoding is encoding:
                        self._frames[topic] = (frame, received_at)
                        self._previews.pop(topic, None)
                previews[topic] = self._preview(topic)
            return CameraPreviewSnapshot(
                rgb=previews[Topic.CAMERA_RGB],
                depth=previews[Topic.CAMERA_DEPTH],
                depth_near_meters=self._config.depth_near_meters,
                depth_far_meters=depth_far,
            )

    def _preview(self, topic: Topic) -> CameraPreview:
        received = self._frames.get(topic)
        if received is None:
            return CameraPreview(connected=False)
        frame, received_at = received
        age = max(0.0, self._clock() - received_at)
        if age > self._config.max_age_seconds:
            return CameraPreview(connected=False, age_seconds=age)
        if topic not in self._previews:
            pixels = (
                frame.data if frame.encoding is CameraEncoding.RGB8 else self._colorize_depth(frame)
            )
            image = Image.fromarray(pixels)
            image.thumbnail((self._config.max_width, self._config.max_width))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self._config.jpeg_quality)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            self._previews[topic] = CameraPreview(
                connected=True,
                image=f"data:image/jpeg;base64,{encoded}",
                width=frame.intrinsics.width,
                height=frame.intrinsics.height,
                frame_id=frame.frame_id,
                frame_number=frame.frame_number,
            )
        return replace(self._previews[topic], age_seconds=age)

    def _colorize_depth(self, frame: CameraImage) -> np.ndarray:
        meters = frame.data.astype(np.float32) * frame.depth_scale
        return colorize_depth(meters, self._config.depth_near_meters, self._depth_far)

    def reset(self) -> None:
        with self._lock:
            self._frames.clear()
            self._previews.clear()
            for topic in (Topic.CAMERA_RGB, Topic.CAMERA_DEPTH):
                self._subscriber.receive_with_timestamp(topic)

    def close(self) -> None:
        self._subscriber.close()
