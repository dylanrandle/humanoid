"""Live managed-node rate monitoring."""

import threading
import time
from collections.abc import Callable, Mapping

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.node import NodeRateSample, NodeRateStatus

MINIMUM_HEALTHY_NODE_RATE_RATIO = 0.9
MAXIMUM_NODE_RATE_AGE_SECONDS = 2.5
NODE_RATE_SUBSCRIBER_QUEUE_SIZE = 256
NODE_RATE_RECEIVE_TIMEOUT_MS = 100
NODE_RATE_THREAD_JOIN_TIMEOUT_SECONDS = 0.5


class NodeRateMonitor:
    """Track target and measured loop rates for active managed node processes."""

    def __init__(
        self,
        subscriber: Subscriber | None = None,
        minimum_healthy_ratio: float = MINIMUM_HEALTHY_NODE_RATE_RATIO,
        max_age_seconds: float = MAXIMUM_NODE_RATE_AGE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < minimum_healthy_ratio <= 1.0:
            raise ValueError(f"{minimum_healthy_ratio=} must be in (0, 1].")
        if max_age_seconds <= 0.0:
            raise ValueError(f"{max_age_seconds=} must be positive.")
        self._subscriber = subscriber or Subscriber(
            topics=[Topic.NODE_RATE],
            queue_size=NODE_RATE_SUBSCRIBER_QUEUE_SIZE,
        )
        self._minimum_healthy_ratio = minimum_healthy_ratio
        self._max_age_seconds = max_age_seconds
        self._clock = clock
        self._samples: dict[int, tuple[NodeRateSample, float]] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._consumer_thread = threading.Thread(
            target=self._consume_samples,
            name="node-rate-monitor",
            daemon=True,
        )
        self._consumer_thread.start()

    def snapshot(self, active_nodes: Mapping[str, int]) -> list[NodeRateStatus]:
        with self._lock:
            now = self._clock()
            active_pids = set(active_nodes.values())
            self._samples = {
                pid: observed for pid, observed in self._samples.items() if pid in active_pids
            }
            return [self._status(node_name, pid, now) for node_name, pid in active_nodes.items()]

    def close(self) -> None:
        self._running.clear()
        self._consumer_thread.join(timeout=NODE_RATE_THREAD_JOIN_TIMEOUT_SECONDS)
        self._subscriber.close()

    def _consume_samples(self) -> None:
        while self._running.is_set():
            sample = self._subscriber.receive(
                Topic.NODE_RATE,
                timeout=NODE_RATE_RECEIVE_TIMEOUT_MS,
            )
            if sample is None:
                continue
            received_at = self._clock()
            with self._lock:
                self._samples[sample.pid] = (sample, received_at)

    def _status(self, node_name: str, pid: int, now: float) -> NodeRateStatus:
        observed = self._samples.get(pid)
        if observed is None or observed[0].node_name != node_name:
            return NodeRateStatus(
                node_name=node_name,
                pid=pid,
                target_rate_hz=None,
                measured_rate_hz=None,
                healthy=False,
                age_seconds=None,
            )

        sample, received_at = observed
        age = now - received_at
        measured_rate_hz = sample.measured_rate_hz or None
        healthy = (
            sample.target_rate_hz > 0.0
            and measured_rate_hz is not None
            and age <= self._max_age_seconds
            and measured_rate_hz >= sample.target_rate_hz * self._minimum_healthy_ratio
        )
        return NodeRateStatus(
            node_name=node_name,
            pid=pid,
            target_rate_hz=round(sample.target_rate_hz, 1),
            measured_rate_hz=(round(measured_rate_hz, 1) if measured_rate_hz is not None else None),
            healthy=healthy,
            age_seconds=round(age, 1),
        )
