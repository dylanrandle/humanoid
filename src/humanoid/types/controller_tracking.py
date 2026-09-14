"""Shared configuration and telemetry types for controller tracking."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TASK_POSITION_SIZE = 3
TASK_QUATERNION_SIZE = 4

ControllerTrackingSegment = Literal[
    "figure_eight",
    "figure_eight_settle",
    "joint_comparison",
    "joint_comparison_settle",
    "cartesian_comparison",
    "cartesian_comparison_settle",
    "joint_home",
    "joint_rest",
    "joint_home_settle",
]
JointCommandStream = Literal["controller", "robot"]
JointTelemetryStream = Literal["controller", "robot", "state"]


@dataclass(frozen=True, kw_only=True)
class ControllerTrackingEndpoint:
    """Corresponding measured joint-space and task-space endpoint."""

    joint_positions_rad: dict[str, float]
    task_frame: str
    task_position_m: tuple[float, float, float]
    task_quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.joint_positions_rad:
            raise ValueError("A tracking endpoint must define joint positions.")
        if any(not name.strip() for name in self.joint_positions_rad):
            raise ValueError("Tracking endpoint joint names must not be empty.")
        if not all(math.isfinite(value) for value in self.joint_positions_rad.values()):
            raise ValueError("Tracking endpoint joint positions must be finite.")
        if not self.task_frame.strip():
            raise ValueError("A tracking endpoint task frame must not be empty.")
        if len(self.task_position_m) != TASK_POSITION_SIZE:
            raise ValueError("Tracking endpoint task position must contain xyz coordinates.")
        if not all(math.isfinite(value) for value in self.task_position_m):
            raise ValueError("Tracking endpoint task position must be finite.")
        if len(self.task_quaternion_wxyz) != TASK_QUATERNION_SIZE:
            raise ValueError("Tracking endpoint task quaternion must contain wxyz coordinates.")
        if not all(math.isfinite(value) for value in self.task_quaternion_wxyz):
            raise ValueError("Tracking endpoint task quaternion must be finite.")
        quaternion_norm = math.sqrt(sum(value**2 for value in self.task_quaternion_wxyz))
        if not math.isclose(quaternion_norm, 1.0, abs_tol=1e-6):
            raise ValueError("Tracking endpoint task quaternion must be normalized.")


@dataclass(frozen=True, kw_only=True)
class ControllerTrackingComparisonConfig:
    """Start and end poses for a joint-versus-Cartesian tracking comparison."""

    start: ControllerTrackingEndpoint
    end: ControllerTrackingEndpoint

    def __post_init__(self) -> None:
        if self.start.task_frame != self.end.task_frame:
            raise ValueError("Tracking comparison endpoints must use the same task frame.")


@dataclass(frozen=True, kw_only=True)
class ControllerCommandTiming:
    """Receiver-observed timing of one joint command in a marked test window."""

    segment: ControllerTrackingSegment
    timestamp_s: float
    stream: JointCommandStream = "controller"
    source_timestamp_s: float | None = None
    window_index: int = 0


@dataclass(frozen=True, kw_only=True)
class NativeJointSample:
    """One losslessly captured command or state message in a marked test window."""

    segment: ControllerTrackingSegment
    window_index: int
    stream: JointTelemetryStream
    received_timestamp_s: float
    source_timestamp_s: float
    joint_names: tuple[str, ...]
    joint_positions_rad: NDArray[np.float64]
    joint_velocities_rad_s: NDArray[np.float64] | None
    tool_position_m: NDArray[np.float64]


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
