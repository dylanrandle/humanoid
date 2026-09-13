"""Data models and constants for controller tracking."""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.types.controller_tracking import (
    ControllerCommandTiming,
    ControllerTrackingSegment,
)
from humanoid.types.homing import HomingPreset
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotJointCommand, RobotState

Plane = Literal["xy", "xz", "yz"]
Segment = ControllerTrackingSegment
GripperBounds = tuple[NDArray[np.float64], NDArray[np.float64]]
JointTarget = tuple[HomingPreset, NDArray[np.float64]]

JOINT_HOME_REST_SEGMENTS = frozenset({"joint_home", "joint_rest"})
FIGURE_EIGHT_SEGMENTS = frozenset({"figure_eight", "figure_eight_settle"})
JOINT_COMPARISON_SEGMENTS = frozenset({"joint_comparison", "joint_comparison_settle"})
CARTESIAN_COMPARISON_SEGMENTS = frozenset({"cartesian_comparison", "cartesian_comparison_settle"})

PLANE_AXES: dict[Plane, tuple[int, int]] = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}
AXIS_NAMES = ("x", "y", "z")

DEFAULT_WIDTH_METERS = 0.08
DEFAULT_HEIGHT_METERS = 0.04
DEFAULT_PERIOD_SECONDS = 8.0
DEFAULT_CYCLES = 3
DEFAULT_COMPARISON_DURATION_SECONDS = 8.0
DEFAULT_JOINT_CYCLES = 2
DEFAULT_RAMP_SECONDS = 2.0
DEFAULT_COMMAND_RATE_HZ = 30.0
DEFAULT_SETTLE_SECONDS = 2.0
DEFAULT_START_DELAY_SECONDS = 3.0
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 5.0
DEFAULT_FEEDBACK_TIMEOUT_SECONDS = 1.0
DEFAULT_HOME_POSITION_TOLERANCE_RAD = 0.03
DEFAULT_HOME_STABLE_SECONDS = 0.3
DEFAULT_HOME_TIMEOUT_SECONDS = 8.0
DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION = 0.05
MAX_GRIPPER_LIMIT_MARGIN_FRACTION = 0.5
MAX_PLOT_POINTS = 2_000
BOUNDS_EPSILON = 1e-12
PLOT_PADDING_FRACTION = 0.08
MIN_PLOT_PADDING = 1e-3
MIN_JOINT_PLOT_SPAN_RAD = float(np.deg2rad(10.0))
MIN_JOINT_VELOCITY_PLOT_SPAN_RAD_S = float(np.deg2rad(20.0))


@dataclass(frozen=True, kw_only=True)
class ControllerTrackingSettings:
    """Validated parameters for one controller-tracking run."""

    plane: Plane = "xz"
    width_m: float = DEFAULT_WIDTH_METERS
    height_m: float = DEFAULT_HEIGHT_METERS
    period_s: float = DEFAULT_PERIOD_SECONDS
    cycles: int = DEFAULT_CYCLES
    comparison_duration_s: float = DEFAULT_COMPARISON_DURATION_SECONDS
    joint_cycles: int = DEFAULT_JOINT_CYCLES
    ramp_s: float = DEFAULT_RAMP_SECONDS
    rate_hz: float = DEFAULT_COMMAND_RATE_HZ
    settle_s: float = DEFAULT_SETTLE_SECONDS
    start_delay_s: float = DEFAULT_START_DELAY_SECONDS
    connection_timeout_s: float = DEFAULT_CONNECTION_TIMEOUT_SECONDS
    feedback_timeout_s: float = DEFAULT_FEEDBACK_TIMEOUT_SECONDS
    home_position_tolerance_rad: float = DEFAULT_HOME_POSITION_TOLERANCE_RAD
    home_stable_s: float = DEFAULT_HOME_STABLE_SECONDS
    home_timeout_s: float = DEFAULT_HOME_TIMEOUT_SECONDS
    move_gripper: bool = True
    gripper_min_rad: float | None = None
    gripper_max_rad: float | None = None
    gripper_period_s: float | None = None
    gripper_limit_margin_fraction: float = DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION

    def __post_init__(self) -> None:
        positive_values = {
            "width": self.width_m,
            "height": self.height_m,
            "period": self.period_s,
            "comparison duration": self.comparison_duration_s,
            "rate": self.rate_hz,
            "connection timeout": self.connection_timeout_s,
            "feedback timeout": self.feedback_timeout_s,
            "home position tolerance": self.home_position_tolerance_rad,
            "home stable duration": self.home_stable_s,
            "home timeout": self.home_timeout_s,
        }
        for label, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
        if self.cycles <= 0:
            raise ValueError("cycles must be positive")
        if self.joint_cycles < 0:
            raise ValueError("joint cycles must be non-negative")
        for label, value in {
            "ramp": self.ramp_s,
            "settle": self.settle_s,
            "start delay": self.start_delay_s,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be non-negative and finite")
        if self.ramp_s > self.duration_s / 2.0:
            raise ValueError("ramp must not exceed half of the trajectory duration")
        self._validate_gripper_settings()

    def _validate_gripper_settings(self) -> None:
        if (self.gripper_min_rad is None) != (self.gripper_max_rad is None):
            raise ValueError("gripper min and max must be provided together")
        if self.gripper_min_rad is not None:
            assert self.gripper_max_rad is not None
            if not math.isfinite(self.gripper_min_rad) or not math.isfinite(self.gripper_max_rad):
                raise ValueError("gripper min and max must be finite")
            if self.gripper_min_rad >= self.gripper_max_rad:
                raise ValueError("gripper min must be less than gripper max")
        if self.gripper_period_s is not None and (
            not math.isfinite(self.gripper_period_s) or self.gripper_period_s <= 0.0
        ):
            raise ValueError("gripper period must be positive and finite")
        if (
            not math.isfinite(self.gripper_limit_margin_fraction)
            or not 0.0 <= self.gripper_limit_margin_fraction < MAX_GRIPPER_LIMIT_MARGIN_FRACTION
        ):
            raise ValueError("gripper limit margin must be finite and in [0, 0.5)")
        if not self.move_gripper and (
            self.gripper_min_rad is not None or self.gripper_period_s is not None
        ):
            raise ValueError("gripper motion options cannot be used with hold gripper")

    @property
    def duration_s(self) -> float:
        return self.period_s * self.cycles

    @property
    def effective_gripper_period_s(self) -> float:
        """Use one gripper cycle per figure-eight cycle unless explicitly overridden."""
        return self.period_s if self.gripper_period_s is None else self.gripper_period_s


@dataclass(frozen=True, kw_only=True)
class TrackingSample:
    """One simultaneous desired-versus-measured controller sample."""

    segment: Segment
    elapsed_s: float
    state_timestamp_s: float
    commanded_position_m: NDArray[np.float64]
    measured_position_m: NDArray[np.float64]
    position_error_m: NDArray[np.float64]
    orientation_error_rad: float
    joint_command_timestamp_s: float
    state_minus_joint_command_s: float
    arm_joint_names: tuple[str, ...]
    commanded_arm_joint_positions_rad: NDArray[np.float64]
    measured_arm_joint_positions_rad: NDArray[np.float64]
    arm_joint_position_errors_rad: NDArray[np.float64]
    commanded_arm_joint_velocities_rad_s: NDArray[np.float64] | None
    measured_arm_joint_velocities_rad_s: NDArray[np.float64]
    gripper_joint_names: tuple[str, ...] = ()
    commanded_gripper_positions_rad: NDArray[np.float64] | None = None
    measured_gripper_positions_rad: NDArray[np.float64] | None = None
    gripper_position_errors_rad: NDArray[np.float64] | None = None
    commanded_gripper_velocities_rad_s: NDArray[np.float64] | None = None
    measured_gripper_velocities_rad_s: NDArray[np.float64] | None = None


@dataclass(frozen=True, kw_only=True)
class ErrorStatistics:
    rms: float
    mean: float
    p95: float
    maximum: float


StatisticsTableRow = tuple[str, ErrorStatistics, float]


@dataclass(frozen=True, kw_only=True)
class TrackingStatistics:
    position_m: ErrorStatistics
    orientation_rad: ErrorStatistics
    arm_joint_position_rad: tuple[ErrorStatistics, ...]
    gripper_position_rad: tuple[ErrorStatistics, ...]


@dataclass(frozen=True, kw_only=True)
class TrackingRun:
    samples: list[TrackingSample]
    completed: bool
    failure_reason: str | None = None
    controller_command_timings: list[ControllerCommandTiming] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class RuntimeFeedback:
    """Latest controller output and hardware feedback with receipt times."""

    state: RobotState
    joint_command: RobotJointCommand
    mode: Mode
    last_state_received_s: float
    last_joint_command_received_s: float
    last_mode_received_s: float


@dataclass(frozen=True, kw_only=True)
class ResolvedTrackingComparison:
    """Validated comparison endpoints in robot configuration and SE(3) form."""

    start_joint_positions: NDArray[np.float64]
    end_joint_positions: NDArray[np.float64]
    start_task_pose: pin.SE3
    end_task_pose: pin.SE3
