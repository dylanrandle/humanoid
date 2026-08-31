"""Sliding-window topic-rate measurements for the operator dashboard."""

from __future__ import annotations

import time
from collections import deque

MINIMUM_RATE_SAMPLE_COUNT = 2


class TopicRate:
    """Track message frequency and freshness over a bounded time window."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def observe(self, timestamp: float | None = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        self._timestamps.append(now)
        self._prune(now)

    def sample(self, timestamp: float | None = None) -> tuple[float, float | None]:
        now = time.monotonic() if timestamp is None else timestamp
        self._prune(now)
        if not self._timestamps:
            return 0.0, None
        age = max(0.0, now - self._timestamps[-1])
        if len(self._timestamps) < MINIMUM_RATE_SAMPLE_COUNT:
            return 0.0, age
        duration = self._timestamps[-1] - self._timestamps[0]
        rate = 0.0 if duration <= 0.0 else (len(self._timestamps) - 1) / duration
        return rate, age

    def _prune(self, timestamp: float) -> None:
        cutoff = timestamp - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
