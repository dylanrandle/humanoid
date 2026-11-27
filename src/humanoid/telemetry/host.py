"""Nonblocking, shared host-resource samples for console requests."""

import threading
import time
from collections.abc import Callable

import psutil

from humanoid.config.monitoring import HOST_RESOURCE_SAMPLE_INTERVAL_SECONDS
from humanoid.types.host import HostResourceStatus


class HostResourceMonitor:
    """Sample all host processes, independent of robot state and request threads."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._sampled_at: float | None = None
        self._cpu_times: dict[str, float] | None = None
        self._status: HostResourceStatus | None = None

    def snapshot(self) -> HostResourceStatus | None:
        with self._lock:
            now = self._clock()
            if (
                self._sampled_at is not None
                and now - self._sampled_at < HOST_RESOURCE_SAMPLE_INTERVAL_SECONDS
            ):
                return self._status
            self._sampled_at = now
            try:
                times = psutil.cpu_times()._asdict()
                memory = psutil.virtual_memory()
            except OSError:
                self._cpu_times = None
                self._status = None
                return None

            cpu_percent = None
            if self._cpu_times is not None:
                # Guest time is already included in user/nice. I/O wait is idle.
                deltas = {
                    name: max(0.0, value - self._cpu_times.get(name, value))
                    for name, value in times.items()
                    if name not in {"guest", "guest_nice"}
                }
                total = sum(deltas.values())
                idle = deltas.get("idle", 0.0) + deltas.get("iowait", 0.0)
                if total > 0.0:
                    cpu_percent = round(100.0 * max(0.0, total - idle) / total, 1)
            self._cpu_times = times
            self._status = HostResourceStatus(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_bytes=memory.total - memory.available,
                memory_total_bytes=memory.total,
            )
            return self._status
