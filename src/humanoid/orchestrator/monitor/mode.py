"""Live orchestrator mode monitoring."""

import threading
import time

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.orchestrator import Mode, ModeStatus


class ModeMonitor:
    """Track the latest mode broadcast by the orchestrator."""

    def __init__(self, subscriber: Subscriber | None = None, max_age_seconds: float = 2.0):
        self._subscriber = subscriber or Subscriber(topics=[Topic.ORCHESTRATOR_MODE])
        self._max_age_seconds = max_age_seconds
        self._mode: Mode | None = None
        self._last_seen_monotonic: float | None = None
        self._lock = threading.Lock()

    def snapshot(self) -> ModeStatus:
        with self._lock:
            message = self._subscriber.receive(Topic.ORCHESTRATOR_MODE)
            if message is not None:
                self._mode = message.mode
                self._last_seen_monotonic = time.monotonic()

            age = (
                time.monotonic() - self._last_seen_monotonic
                if self._last_seen_monotonic is not None
                else None
            )
            connected = age is not None and age <= self._max_age_seconds
            return ModeStatus(
                mode=self._mode if connected else None,
                connected=connected,
                age_seconds=round(age, 1) if age is not None else None,
            )

    def reset(self) -> None:
        with self._lock:
            self._mode = None
            self._last_seen_monotonic = None
            while self._subscriber.receive(Topic.ORCHESTRATOR_MODE) is not None:
                pass

    def close(self) -> None:
        self._subscriber.close()
