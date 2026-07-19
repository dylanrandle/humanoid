"""Live orchestrator-mode monitoring."""

import threading
import time
from collections.abc import Callable, Mapping

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.orchestrator.constants import LOGGING_ACKNOWLEDGEMENT_TIMEOUT_SECONDS
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.node import NodeRateSample, NodeRateStatus
from humanoid.types.orchestrator import Mode, ModeStatus

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


class OrchestratorMonitor:
    """Tracks the latest mode broadcast by the orchestrator."""

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


class LoggingMonitor:
    """Tracks the latest lifecycle report from RobotLoggerNode."""

    def __init__(
        self,
        subscriber: Subscriber | None = None,
        acknowledgement_timeout_seconds: float = LOGGING_ACKNOWLEDGEMENT_TIMEOUT_SECONDS,
    ):
        self._subscriber = subscriber or Subscriber(topics=[Topic.LOGGING_STATUS])
        self._acknowledgement_timeout_seconds = acknowledgement_timeout_seconds
        self._status = self._stopped_status()
        self._pending_since_monotonic: float | None = None
        self._lock = threading.Lock()

    def snapshot(self) -> LoggingStatus:
        with self._lock:
            received = False
            while (status := self._subscriber.receive(Topic.LOGGING_STATUS)) is not None:
                self._status = status
                received = True
            if received:
                if self._status.state in {LoggingState.STARTING, LoggingState.STOPPING}:
                    self._pending_since_monotonic = time.monotonic()
                else:
                    self._pending_since_monotonic = None
            elif self._pending_request_expired():
                action = "start" if self._status.state is LoggingState.STARTING else "stop"
                self._status = LoggingStatus(
                    timestamp=time.time(),
                    state=LoggingState.FAILED,
                    error=f"Data logging did not acknowledge the {action} request.",
                )
                self._pending_since_monotonic = None
            return self._status

    def start_requested(self) -> None:
        self._set_pending(LoggingState.STARTING)

    def stop_requested(self) -> None:
        self._set_pending(LoggingState.STOPPING)

    def fail(self, error: str) -> None:
        with self._lock:
            self._pending_since_monotonic = None
            self._status = LoggingStatus(
                timestamp=time.time(),
                state=LoggingState.FAILED,
                error=error,
            )

    def reset(self) -> None:
        with self._lock:
            self._status = self._stopped_status()
            self._pending_since_monotonic = None
            while self._subscriber.receive(Topic.LOGGING_STATUS) is not None:
                pass

    def close(self) -> None:
        self._subscriber.close()

    def _set_pending(self, state: LoggingState) -> None:
        with self._lock:
            while self._subscriber.receive(Topic.LOGGING_STATUS) is not None:
                pass
            self._status = LoggingStatus(timestamp=time.time(), state=state)
            self._pending_since_monotonic = time.monotonic()

    def _pending_request_expired(self) -> bool:
        return (
            self._pending_since_monotonic is not None
            and time.monotonic() - self._pending_since_monotonic
            >= self._acknowledgement_timeout_seconds
        )

    @staticmethod
    def _stopped_status() -> LoggingStatus:
        return LoggingStatus(timestamp=time.time(), state=LoggingState.STOPPED)
