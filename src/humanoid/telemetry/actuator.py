"""Live actuator-health monitoring for the operator console."""

import threading
import time
from collections.abc import Callable

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.actuator import ActuatorHealthReport, ActuatorHealthStatus

MAXIMUM_ACTUATOR_HEALTH_AGE_SECONDS = 2.5


class ActuatorHealthMonitor:
    """Retain the latest actuator report, including faults after a driver exits."""

    def __init__(
        self,
        subscriber: Subscriber | None = None,
        max_age_seconds: float = MAXIMUM_ACTUATOR_HEALTH_AGE_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if max_age_seconds <= 0.0:
            raise ValueError(f"{max_age_seconds=} must be positive.")
        self._subscriber = subscriber or Subscriber(topics=[Topic.ACTUATOR_HEALTH])
        self._max_age_seconds = max_age_seconds
        self._clock = clock
        self._report: ActuatorHealthReport | None = None
        self._lock = threading.Lock()

    def snapshot(self) -> ActuatorHealthStatus:
        with self._lock:
            while (report := self._subscriber.receive(Topic.ACTUATOR_HEALTH)) is not None:
                self._report = report

            report = self._report
            age = max(0.0, self._clock() - report.timestamp) if report is not None else None
            connected = age is not None and age <= self._max_age_seconds
            actuators = report.actuators if report is not None else ()
            error = report.error if report is not None else None
            healthy = (
                connected
                and error is None
                and bool(actuators)
                and all(actuator.healthy for actuator in actuators)
            )
            return ActuatorHealthStatus(
                connected=connected,
                healthy=healthy,
                age_seconds=round(age, 1) if age is not None else None,
                actuators=actuators,
                error=error,
            )

    def reset(self) -> None:
        with self._lock:
            self._report = None
            while self._subscriber.receive(Topic.ACTUATOR_HEALTH) is not None:
                pass

    def close(self) -> None:
        self._subscriber.close()
