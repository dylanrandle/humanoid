"""Thread-safe stream telemetry, independent of WebRTC session ownership."""

import threading
from dataclasses import replace

from humanoid.types.streaming import StreamMetrics


class StreamTelemetry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: dict[str, StreamMetrics] = {}

    def register(self, session_id: str, camera: str, stream: str) -> None:
        with self._lock:
            self._streams[session_id] = StreamMetrics(camera, stream, "new")

    def connection(self, session_id: str, state: str) -> None:
        with self._lock:
            previous = self._streams.get(session_id)
            if previous is not None:
                self._streams[session_id] = replace(previous, connection_state=state)

    def frame(self, session_id: str, age: float | None) -> None:
        with self._lock:
            previous = self._streams.get(session_id)
            if previous is not None:
                self._streams[session_id] = replace(
                    previous,
                    frames_sent=previous.frames_sent + 1,
                    frame_age_seconds=age,
                )

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._streams.pop(session_id, None)

    def snapshot(self) -> dict[str, StreamMetrics]:
        with self._lock:
            return dict(self._streams)
