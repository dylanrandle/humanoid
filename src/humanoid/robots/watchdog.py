"""Shared command-loss watchdog for velocity-controlled robot actuators."""

import math
import time
from collections.abc import Callable


class VelocityCommandWatchdog:
    """Stop velocity actuators once when an active command becomes stale."""

    def __init__(
        self,
        stop: Callable[[], None],
        timeout_seconds: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("Robot command timeout must be positive and finite.")
        self._stop = stop
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._last_command_time: float | None = None
        self._velocity_active = False

    def observe_command(self, *, velocity_active: bool) -> None:
        self._last_command_time = self._clock()
        self._velocity_active = velocity_active

    def stop_if_stale(self) -> bool:
        if (
            self._last_command_time is None
            or not self._velocity_active
            or self._clock() - self._last_command_time < self._timeout_seconds
        ):
            return False
        self._stop()
        self._velocity_active = False
        return True

    def stop(self) -> None:
        """Stop immediately, including during node shutdown."""

        self._stop()
        self._velocity_active = False
