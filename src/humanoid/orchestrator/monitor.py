"""Live orchestrator-mode monitoring."""

import threading
import time

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.orchestrator.constants import LOGGING_ACKNOWLEDGEMENT_TIMEOUT_SECONDS
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.orchestrator import Mode, ModeStatus


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
