"""Measure channel publish rates from raw LCM traffic without decoding payloads."""

import math
import threading
import time
from collections import deque
from collections.abc import Callable

import lcm

from humanoid.config.telemetry import LCM_RATE_QUEUE_CAPACITY, LCM_RATE_WINDOW_SECONDS
from humanoid.constants import DEFAULT_LCM_URL
from humanoid.types.telemetry import LCMChannelStatus


class LCMRateMonitor:
    """Passively observe all channels on the application's bus, including cameras.

    ``url`` selects the bus and ``window_seconds`` controls the rolling rate window.
    Rates count received messages from all publishers on each channel, divided by
    the window duration (or elapsed monitoring time during startup). They measure
    observed traffic, not guaranteed delivery or compliance with a target rate.
    Payloads are not decoded; channel names are discovered from received traffic.
    """

    def __init__(
        self,
        url: str = DEFAULT_LCM_URL,
        *,
        window_seconds: float = LCM_RATE_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive.")
        self._window_seconds = window_seconds
        self._clock = clock
        self._started_at = clock()
        self._timestamps: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._lc = lcm.LCM(url)
        self._subscription = self._lc.subscribe(".*", self._observe)
        self._subscription.set_queue_capacity(LCM_RATE_QUEUE_CAPACITY)
        self._thread = threading.Thread(target=self._consume, name="lcm-rate-monitor", daemon=True)
        self._thread.start()

    def _consume(self) -> None:
        while not self._stop.is_set():
            self._lc.handle_timeout(100)

    def _observe(self, channel: str, _data: bytes) -> None:
        with self._lock:
            now = self._clock()
            timestamps = self._timestamps.setdefault(channel, deque())
            timestamps.append(now)
            self._prune(timestamps, now)

    def _prune(self, timestamps: deque[float], now: float) -> None:
        # Retain the last receipt even after it expires, to report idle-channel age.
        while len(timestamps) > 1 and timestamps[0] <= now - self._window_seconds:
            timestamps.popleft()

    def snapshot(self) -> list[LCMChannelStatus]:
        """Return rates and receipt ages; channels with no recent traffic stay idle."""
        with self._lock:
            now = self._clock()
            duration = min(self._window_seconds, now - self._started_at)
            statuses = []
            for channel, timestamps in sorted(self._timestamps.items()):
                self._prune(timestamps, now)
                count = len(timestamps) - int(timestamps[0] <= now - self._window_seconds)
                statuses.append(
                    LCMChannelStatus(
                        channel=channel,
                        publish_rate_hz=round(count / duration, 1) if duration > 0 else 0.0,
                        age_seconds=round(now - timestamps[-1], 1),
                        active=count > 0,
                    )
                )
            return statuses

    def reset(self) -> None:
        """Clear channel history and restart measurement after runtime changes."""
        with self._lock:
            self._timestamps.clear()
            self._started_at = self._clock()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._lc.unsubscribe(self._subscription)
