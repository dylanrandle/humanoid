"""Shared loop-rate telemetry for every node process."""

import math
import os
import time
from collections.abc import Callable

from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.types.node import NodeRateSample

DEFAULT_RATE_REPORT_INTERVAL_SECONDS = 1.0


class NodeRateReporter:
    """Measure loop cadence and publish one rolling rate sample per interval."""

    def __init__(  # noqa: PLR0913 - timing and transport dependencies are injectable
        self,
        node_name: str,
        target_rate_hz: float,
        *,
        report_interval_seconds: float = DEFAULT_RATE_REPORT_INTERVAL_SECONDS,
        publisher: Publisher | None = None,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        pid: int | None = None,
    ) -> None:
        if not math.isfinite(target_rate_hz) or target_rate_hz <= 0.0:
            raise ValueError(f"{target_rate_hz=} must be positive and finite.")
        if not math.isfinite(report_interval_seconds) or report_interval_seconds <= 0.0:
            raise ValueError(f"{report_interval_seconds=} must be positive and finite.")
        self.node_name = node_name
        self.target_rate_hz = target_rate_hz
        self.report_interval_seconds = report_interval_seconds
        self.publisher = publisher or Publisher()
        self._clock = clock
        self._wall_clock = wall_clock
        self._pid = pid if pid is not None else os.getpid()
        self._window_started: float | None = None
        self._iterations = 0

    def start(self) -> None:
        """Announce the target immediately; measured cadence follows after one window."""
        self._window_started = None
        self._iterations = 0
        self._publish(measured_rate_hz=0.0)

    def observe_iteration(self) -> None:
        """Record one loop start and publish when the measurement window is full."""
        now = self._clock()
        if self._window_started is None:
            self._window_started = now
            return

        self._iterations += 1
        elapsed = now - self._window_started
        if elapsed < self.report_interval_seconds:
            return

        self._publish(measured_rate_hz=self._iterations / elapsed)
        self._window_started = now
        self._iterations = 0

    def _publish(self, measured_rate_hz: float) -> None:
        self.publisher.publish(
            NodeRateSample(
                timestamp=self._wall_clock(),
                node_name=self.node_name,
                pid=self._pid,
                target_rate_hz=self.target_rate_hz,
                measured_rate_hz=measured_rate_hz,
            ),
            topic=Topic.NODE_RATE,
        )
