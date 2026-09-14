"""Lossless capture of native joint telemetry and command timing."""

import math
import time
from collections.abc import Callable

import numpy as np

from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.robots.base import Robot
from humanoid.types.controller_tracking import (
    ControllerCommandTiming,
    ControllerPublicationStatistics,
    ControllerTrackingSegment,
    JointCommandStream,
    JointTelemetryStream,
    NativeJointSample,
)
from humanoid.types.robot import RobotJointCommand, RobotState

MINIMUM_PUBLICATION_COUNT = 2
JOINT_COMMAND_STREAM_TOPICS: tuple[tuple[JointCommandStream, Topic], ...] = (
    ("controller", Topic.CONTROLLER_JOINT_COMMAND),
    ("robot", Topic.ROBOT_JOINT_COMMAND),
)
JOINT_STATE_STREAM_TOPIC: tuple[JointTelemetryStream, Topic] = ("state", Topic.ROBOT_STATE)


class ControllerCommandTimingRecorder:
    """Capture source commands, routed commands, and robot state without sampling."""

    def __init__(
        self,
        subscriber: Subscriber | None = None,
        clock: Callable[[], float] = time.perf_counter,
        robot: Robot | None = None,
    ) -> None:
        # This subscriber is deliberately separate from the latest-value runtime
        # subscriber. Per-topic unbounded queues preserve source commands, final
        # routed commands, and state feedback until shutdown.
        self._stream_topics: tuple[tuple[JointTelemetryStream, Topic], ...] = (
            (*JOINT_COMMAND_STREAM_TOPICS, JOINT_STATE_STREAM_TOPIC)
            if robot is not None
            else JOINT_COMMAND_STREAM_TOPICS
        )
        self._subscriber = subscriber or Subscriber(
            topics=[topic for _, topic in self._stream_topics],
            queue_size=None,
            clock=clock,
        )
        self._clock = clock
        self._robot = robot
        self._windows: list[tuple[int, ControllerTrackingSegment, str, float, float]] = []
        self._active_window: tuple[ControllerTrackingSegment, str, float] | None = None
        self._native_joint_samples: list[NativeJointSample] = []
        self._closed = False

    def begin(self, segment: ControllerTrackingSegment, setting: str) -> None:
        """Start attributing raw controller publications to ``segment``."""
        if self._closed:
            raise RuntimeError("Controller command timing recorder is closed")
        if self._active_window is not None:
            raise RuntimeError("Controller command timing window is already active")
        self._active_window = (segment, setting, self._clock())

    def end(self) -> None:
        """Finish the active publication timing window."""
        if self._active_window is None:
            return
        segment, setting, started_s = self._active_window
        self._windows.append((len(self._windows), segment, setting, started_s, self._clock()))
        self._active_window = None

    @property
    def native_joint_samples(self) -> list[NativeJointSample]:
        """Return converted native-rate samples after :meth:`close`."""
        if not self._closed:
            raise RuntimeError("Controller command timing recorder is still open")
        return list(self._native_joint_samples)

    def close(self) -> list[ControllerCommandTiming]:
        """Stop capture and return publications that occurred inside marked windows."""
        if self._closed:
            return []
        self.end()
        # Stop the receive thread before draining so publications already being
        # decoded cannot race the final queue read.
        self._subscriber.close()
        messages = []
        for stream, topic in self._stream_topics:
            while (received := self._subscriber.receive_with_timestamp(topic)) is not None:
                message, received_at_s = received
                messages.append((stream, topic, message, received_at_s))
        self._closed = True

        timings = []
        native_samples = []
        for stream, _topic, message, received_at_s in messages:
            for window_index, segment, setting, started_s, ended_s in self._windows:
                if started_s <= received_at_s <= ended_s:
                    if stream != "state":
                        timings.append(
                            ControllerCommandTiming(
                                segment=segment,
                                setting=setting,
                                timestamp_s=received_at_s,
                                stream=stream,
                                source_timestamp_s=message.timestamp,
                                window_index=window_index,
                            )
                        )
                    if self._robot is not None:
                        native_samples.append(
                            _native_joint_sample(
                                robot=self._robot,
                                segment=segment,
                                setting=setting,
                                window_index=window_index,
                                stream=stream,
                                message=message,
                                received_at_s=received_at_s,
                            )
                        )
                    break
        self._native_joint_samples = sorted(
            native_samples,
            key=lambda sample: (sample.received_timestamp_s, sample.stream),
        )
        return sorted(timings, key=lambda timing: (timing.timestamp_s, timing.stream))


def _native_joint_sample(  # noqa: PLR0913 - preserves one complete source message
    *,
    robot: Robot,
    segment: ControllerTrackingSegment,
    setting: str,
    window_index: int,
    stream: JointTelemetryStream,
    message: RobotJointCommand | RobotState,
    received_at_s: float,
) -> NativeJointSample:
    """Convert generalized coordinates to named arm/gripper telemetry."""
    joint_indices = (*robot.get_arm_joint_indices(), *robot.get_gripper_joint_indices())
    joint_names = tuple(robot.joint_idx_to_name(index) for index in joint_indices)
    positions = np.array(
        [robot.joint_position_from_q(message.joint_positions, index) for index in joint_indices]
    )
    velocity_indices = robot.get_joint_velocity_indices(list(joint_indices))
    velocities = (
        message.joint_velocities[velocity_indices].copy()
        if message.joint_velocities is not None
        else None
    )
    return NativeJointSample(
        segment=segment,
        setting=setting,
        window_index=window_index,
        stream=stream,
        received_timestamp_s=received_at_s,
        source_timestamp_s=message.timestamp,
        joint_names=joint_names,
        joint_positions_rad=positions,
        joint_velocities_rad_s=velocities,
        tool_position_m=robot.get_tool_command_pose(message.joint_positions).translation.copy(),
    )


def controller_publication_statistics(
    timings: list[ControllerCommandTiming],
    target_rate_hz: float,
) -> ControllerPublicationStatistics:
    """Summarize one stream without treating gaps between test windows as periods."""
    if not math.isfinite(target_rate_hz) or target_rate_hz <= 0.0:
        raise ValueError("Target publication rate must be positive and finite")
    if len(timings) < MINIMUM_PUBLICATION_COUNT:
        raise ValueError("At least two controller publications are required")
    if len({timing.stream for timing in timings}) != 1:
        raise ValueError("Publication timings must contain exactly one command stream")

    periods_by_window = []
    for window_index in sorted({timing.window_index for timing in timings}):
        timestamps = np.array(
            sorted(timing.timestamp_s for timing in timings if timing.window_index == window_index)
        )
        if timestamps.size > 1:
            periods_by_window.append(np.diff(timestamps))
    if not periods_by_window:
        raise ValueError("At least one timing window must contain two publications")
    periods = np.concatenate(periods_by_window)
    if np.any(periods <= 0.0):
        raise ValueError("Controller publication timestamps must be unique and increasing")
    delayed_threshold_s = 1.5 / target_rate_hz
    return ControllerPublicationStatistics(
        command_count=len(timings),
        mean_rate_hz=float(1.0 / np.mean(periods)),
        median_period_s=float(np.median(periods)),
        p95_period_s=float(np.percentile(periods, 95)),
        maximum_period_s=float(np.max(periods)),
        period_jitter_s=float(np.std(periods)),
        delayed_interval_count=int(np.count_nonzero(periods > delayed_threshold_s)),
    )
