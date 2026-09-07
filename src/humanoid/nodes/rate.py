"""Shared loop-rate telemetry for every node process."""

import math
import os
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path

from humanoid.constants import Topic
from humanoid.middleware.publisher import Publisher
from humanoid.types.node import NodeRateSample

DEFAULT_RATE_REPORT_INTERVAL_SECONDS = 1.0
BYTES_PER_MEBIBYTE = 1024**2
PROC_SELF_STATM = Path("/proc/self/statm")


def read_process_memory_rss_mb() -> float:
    """Return current resident memory on Linux and peak resident memory elsewhere."""
    try:
        statm_fields = PROC_SELF_STATM.read_text().split()
        resident_pages = int(statm_fields[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / BYTES_PER_MEBIBYTE
    except (IndexError, OSError, ValueError):
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_rss_bytes = max_rss if sys.platform == "darwin" else max_rss * 1024
        return max_rss_bytes / BYTES_PER_MEBIBYTE


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
        process_clock: Callable[[], float] = time.process_time,
        memory_reader: Callable[[], float] = read_process_memory_rss_mb,
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
        self._process_clock = process_clock
        self._memory_reader = memory_reader
        self._pid = pid if pid is not None else os.getpid()
        self._window_started: float | None = None
        self._process_cpu_started: float | None = None
        self._iterations = 0

    def start(self) -> None:
        """Announce the target immediately; measured cadence follows after one window."""
        self._window_started = None
        self._process_cpu_started = None
        self._iterations = 0
        self._publish(measured_rate_hz=0.0, cpu_percent=0.0)

    def observe_iteration(self) -> None:
        """Record one loop start and publish when the measurement window is full."""
        now = self._clock()
        if self._window_started is None:
            self._window_started = now
            self._process_cpu_started = self._process_clock()
            return

        self._iterations += 1
        elapsed = now - self._window_started
        if elapsed < self.report_interval_seconds:
            return

        assert self._process_cpu_started is not None
        process_cpu_now = self._process_clock()
        cpu_percent = max(0.0, process_cpu_now - self._process_cpu_started) / elapsed * 100.0
        self._publish(
            measured_rate_hz=self._iterations / elapsed,
            cpu_percent=cpu_percent,
        )
        self._window_started = now
        self._process_cpu_started = process_cpu_now
        self._iterations = 0

    def _publish(self, measured_rate_hz: float, cpu_percent: float) -> None:
        self.publisher.publish(
            NodeRateSample(
                timestamp=self._wall_clock(),
                node_name=self.node_name,
                pid=self._pid,
                target_rate_hz=self.target_rate_hz,
                measured_rate_hz=measured_rate_hz,
                cpu_percent=cpu_percent,
                memory_rss_mb=self._memory_reader(),
            ),
            topic=Topic.NODE_RATE,
        )
