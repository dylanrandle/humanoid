"""Shared telemetry types for controller tracking."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from humanoid.types.actuator import ActuatorEffortSource, ActuatorEffortUnit

ControllerTrackingSegment = Literal[
    "figure_eight",
    "figure_eight_settle",
]
JointCommandStream = Literal["controller", "robot"]
JointTelemetryStream = Literal["controller", "robot", "state"]


@dataclass(frozen=True, kw_only=True)
class ActuatorEffortTrace:
    """One diagnostic setting's signed effort samples in joint order and SI units."""

    joint_names: tuple[str, ...]
    times_s: NDArray[np.float64]
    efforts: NDArray[np.float64]
    units: tuple[ActuatorEffortUnit, ...]
    source: str
    segment_boundaries_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.times_s.ndim != 1 or self.efforts.shape != (
            len(self.times_s),
            len(self.joint_names),
        ):
            raise ValueError("Effort values must match the timestamps and joint names.")
        if len(self.units) != len(self.joint_names):
            raise ValueError("Effort units must match the joint names.")
        if not np.isfinite(self.times_s).all() or np.any(np.diff(self.times_s) < 0.0):
            raise ValueError("Effort timestamps must be finite and ordered.")


@dataclass(frozen=True, kw_only=True)
class ControllerCommandTiming:
    """Receiver-observed timing of one joint command in a marked test window."""

    segment: ControllerTrackingSegment
    setting: str
    timestamp_s: float
    stream: JointCommandStream = "controller"
    source_timestamp_s: float | None = None
    window_index: int = 0


@dataclass(frozen=True, kw_only=True)
class NativeJointSample:
    """One losslessly captured command or state message in a marked test window."""

    segment: ControllerTrackingSegment
    setting: str
    window_index: int
    stream: JointTelemetryStream
    received_timestamp_s: float
    source_timestamp_s: float
    joint_names: tuple[str, ...]
    joint_positions_rad: NDArray[np.float64]
    joint_velocities_rad_s: NDArray[np.float64] | None
    joint_efforts: NDArray[np.float64] | None = None
    tool_position_m: NDArray[np.float64]
    effort_source: ActuatorEffortSource | None = None


@dataclass(frozen=True, kw_only=True)
class JointSmoothnessStatistics:
    """Motion-localized and post-motion smoothness measures for one joint."""

    high_frequency_rms_rad: float
    high_frequency_peak_to_peak_rad: float
    acceleration_p95_rad_s2: float
    jerk_rms_rad_s3: float
    command_acceleration_rms_rad_s2: float | None
    command_acceleration_p95_rad_s2: float | None
    acceleration_limit_fraction: float | None
    settle_peak_to_peak_rad: float | None
    dominant_frequency_hz: float | None


@dataclass(frozen=True, kw_only=True)
class MotionTrace:
    """Uniformly sampled, offline-filtered signals for one native-rate window."""

    segment: ControllerTrackingSegment
    setting: str
    window_index: int
    stream: JointTelemetryStream
    times_s: NDArray[np.float64]
    source_times_s: NDArray[np.float64]
    joint_names: tuple[str, ...]
    positions_rad: NDArray[np.float64]
    filtered_positions_rad: NDArray[np.float64]
    high_frequency_positions_rad: NDArray[np.float64]
    raw_velocities_rad_s: NDArray[np.float64]
    source_velocities_rad_s: NDArray[np.float64] | None
    velocities_rad_s: NDArray[np.float64]
    accelerations_rad_s2: NDArray[np.float64]
    jerks_rad_s3: NDArray[np.float64]
    frequencies_hz: NDArray[np.float64]
    position_psd_rad2_hz: NDArray[np.float64]
    tool_high_frequency_position_m: NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class SmoothnessAnalysis:
    """Native-rate traces and joint metrics for one diagnostic phase."""

    joint_names: tuple[str, ...]
    statistics: tuple[JointSmoothnessStatistics, ...]
    state_traces: tuple[MotionTrace, ...]
    command_traces: tuple[MotionTrace, ...]
    shake_cutoff_hz: float
    tool_high_frequency_rms_m: float | None
    tool_high_frequency_peak_to_peak_m: float | None


@dataclass(frozen=True, kw_only=True)
class ControllerPublicationStatistics:
    """Observed timing statistics for a sequence of OSC joint commands."""

    command_count: int
    mean_rate_hz: float
    median_period_s: float
    p95_period_s: float
    maximum_period_s: float
    period_jitter_s: float
    delayed_interval_count: int
