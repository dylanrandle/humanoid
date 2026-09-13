"""Lossless capture and analysis of raw OSC joint-command publication timing."""

import math
import time
from collections.abc import Callable

import numpy as np

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.controller_tracking import (
    ControllerCommandTiming,
    ControllerPublicationStatistics,
    ControllerTrackingSegment,
)

MINIMUM_PUBLICATION_COUNT = 2


class ControllerCommandTimingRecorder:
    """Capture every raw controller publication within explicitly marked phases."""

    def __init__(
        self,
        subscriber: Subscriber | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        # This subscriber is deliberately separate from the latest-value runtime
        # subscriber. An unbounded queue preserves every controller publication
        # until the diagnostic drains it at shutdown.
        self._subscriber = subscriber or Subscriber(
            topics=[Topic.CONTROLLER_JOINT_COMMAND],
            queue_size=None,
        )
        self._clock = clock
        self._windows: list[tuple[ControllerTrackingSegment, float, float]] = []
        self._active_window: tuple[ControllerTrackingSegment, float] | None = None
        self._closed = False

    def begin(self, segment: ControllerTrackingSegment) -> None:
        """Start attributing raw controller publications to ``segment``."""
        if self._closed:
            raise RuntimeError("Controller command timing recorder is closed")
        if self._active_window is not None:
            raise RuntimeError("Controller command timing window is already active")
        self._active_window = (segment, self._clock())

    def end(self) -> None:
        """Finish the active publication timing window."""
        if self._active_window is None:
            return
        segment, started_s = self._active_window
        self._windows.append((segment, started_s, self._clock()))
        self._active_window = None

    def close(self) -> list[ControllerCommandTiming]:
        """Stop capture and return publications that occurred inside marked windows."""
        if self._closed:
            return []
        self.end()
        messages = []
        while (message := self._subscriber.receive(Topic.CONTROLLER_JOINT_COMMAND)) is not None:
            messages.append(message)
        self._subscriber.close()
        self._closed = True

        timings = []
        for message in messages:
            for segment, started_s, ended_s in self._windows:
                if started_s <= message.timestamp <= ended_s:
                    timings.append(
                        ControllerCommandTiming(
                            segment=segment,
                            timestamp_s=message.timestamp,
                        )
                    )
                    break
        return sorted(timings, key=lambda timing: timing.timestamp_s)


def controller_publication_statistics(
    timings: list[ControllerCommandTiming],
    target_rate_hz: float,
) -> ControllerPublicationStatistics:
    """Summarize achieved rate and period jitter for one timing sequence."""
    if not math.isfinite(target_rate_hz) or target_rate_hz <= 0.0:
        raise ValueError("Target publication rate must be positive and finite")
    if len(timings) < MINIMUM_PUBLICATION_COUNT:
        raise ValueError("At least two controller publications are required")

    timestamps = np.array(sorted(timing.timestamp_s for timing in timings))
    periods = np.diff(timestamps)
    if np.any(periods <= 0.0):
        raise ValueError("Controller publication timestamps must be unique and increasing")
    delayed_threshold_s = 1.5 / target_rate_hz
    return ControllerPublicationStatistics(
        command_count=len(timings),
        mean_rate_hz=float((len(timings) - 1) / (timestamps[-1] - timestamps[0])),
        median_period_s=float(np.median(periods)),
        p95_period_s=float(np.percentile(periods, 95)),
        maximum_period_s=float(np.max(periods)),
        period_jitter_s=float(np.std(periods)),
        delayed_interval_count=int(np.count_nonzero(periods > delayed_threshold_s)),
    )
