"""Configuration shapes for robot-specific controller-tracking comparisons."""

import math
from dataclasses import dataclass
from typing import Literal

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
]


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
    """Timestamp of one raw OSC joint-command publication during a test phase."""

    segment: ControllerTrackingSegment
    timestamp_s: float


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
