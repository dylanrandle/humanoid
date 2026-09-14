"""Data models and constants for controller tracking."""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from humanoid.types.controller_tracking import (
    ControllerCommandTiming,
    ControllerTrackingSegment,
    NativeJointSample,
)
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotJointCommand, RobotState

Plane = Literal["xy", "xz", "yz"]
Segment = ControllerTrackingSegment
GripperBounds = tuple[NDArray[np.float64], NDArray[np.float64]]

FIGURE_EIGHT_SEGMENTS = frozenset({"figure_eight", "figure_eight_settle"})
FIGURE_EIGHT_LOOPS_PER_SETTING = 2
FIGURE_EIGHT_PLANES: tuple[Plane, ...] = ("xy", "xz", "yz")
FIGURE_EIGHT_SIZE_MULTIPLIERS = (1.0, 1.5, 2.0)

PLANE_AXES: dict[Plane, tuple[int, int]] = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}
AXIS_NAMES = ("x", "y", "z")

DEFAULT_WIDTH_METERS = 0.04
DEFAULT_HEIGHT_METERS = 0.02
DEFAULT_PERIOD_SECONDS = 8.0
DEFAULT_ORIENTATION_BIAS_RADIANS = float(np.deg2rad(10.0))
FIGURE_EIGHT_TRANSITION_PERIOD_FRACTION = 0.25
DEFAULT_COMMAND_RATE_HZ = 30.0
DEFAULT_SETTLE_SECONDS = 2.0
DEFAULT_START_DELAY_SECONDS = 3.0
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 5.0
DEFAULT_FEEDBACK_TIMEOUT_SECONDS = 1.0
DEFAULT_GRIPPER_PERIOD_SECONDS = 16.0
DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION = 0.05
MAX_GRIPPER_LIMIT_MARGIN_FRACTION = 0.5
MAX_ORIENTATION_BIAS_RADIANS = float(np.pi / 2.0)
MAX_PLOT_POINTS = 2_000
BOUNDS_EPSILON = 1e-12
PLOT_PADDING_FRACTION = 0.08
MIN_PLOT_PADDING = 1e-3
MIN_JOINT_PLOT_SPAN_RAD = float(np.deg2rad(10.0))
MIN_JOINT_VELOCITY_PLOT_SPAN_RAD_S = float(np.deg2rad(20.0))
DEFAULT_SHAKE_CUTOFF_HZ = 2.0
DEFAULT_DERIVATIVE_SMOOTHING_SECONDS = 0.25
DEFAULT_MAXIMUM_SPECTRUM_HZ = 15.0


@dataclass(frozen=True)
class FigureEightSetting:
    """One plane and size combination in the Cartesian tracking matrix."""

    plane: Plane
    size_multiplier: float

    @property
    def name(self) -> str:
        scale = f"{self.size_multiplier:g}".replace(".", "p")
        return f"{self.plane}_{scale}x"


FIGURE_EIGHT_SETTINGS = tuple(
    FigureEightSetting(plane=plane, size_multiplier=size)
    for size in FIGURE_EIGHT_SIZE_MULTIPLIERS
    for plane in FIGURE_EIGHT_PLANES
)
FIGURE_EIGHT_TOTAL_LOOPS = FIGURE_EIGHT_LOOPS_PER_SETTING * len(FIGURE_EIGHT_SETTINGS)


@dataclass(frozen=True, kw_only=True)
class ControllerTrackingSettings:
    """Validated parameters for one Cartesian figure-eight run."""

    width_m: float = DEFAULT_WIDTH_METERS
    height_m: float = DEFAULT_HEIGHT_METERS
    period_s: float = DEFAULT_PERIOD_SECONDS
    orientation_bias_rad: float = DEFAULT_ORIENTATION_BIAS_RADIANS
    rate_hz: float = DEFAULT_COMMAND_RATE_HZ
    settle_s: float = DEFAULT_SETTLE_SECONDS
    start_delay_s: float = DEFAULT_START_DELAY_SECONDS
    connection_timeout_s: float = DEFAULT_CONNECTION_TIMEOUT_SECONDS
    feedback_timeout_s: float = DEFAULT_FEEDBACK_TIMEOUT_SECONDS
    move_gripper: bool = True
    gripper_min_rad: float | None = None
    gripper_max_rad: float | None = None
    gripper_period_s: float = DEFAULT_GRIPPER_PERIOD_SECONDS
    gripper_limit_margin_fraction: float = DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION
    velocity_feedforward: bool = True
    shake_cutoff_hz: float = DEFAULT_SHAKE_CUTOFF_HZ
    derivative_smoothing_s: float = DEFAULT_DERIVATIVE_SMOOTHING_SECONDS

    def __post_init__(self) -> None:
        positive_values = {
            "width": self.width_m,
            "height": self.height_m,
            "period": self.period_s,
            "rate": self.rate_hz,
            "connection timeout": self.connection_timeout_s,
            "feedback timeout": self.feedback_timeout_s,
            "shake cutoff": self.shake_cutoff_hz,
            "derivative smoothing duration": self.derivative_smoothing_s,
        }
        for label, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
        for label, value in {
            "settle": self.settle_s,
            "start delay": self.start_delay_s,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be non-negative and finite")
        if (
            not math.isfinite(self.orientation_bias_rad)
            or not 0.0 <= self.orientation_bias_rad <= MAX_ORIENTATION_BIAS_RADIANS
        ):
            raise ValueError("orientation bias must be finite and in [0, pi/2] radians")
        if self.shake_cutoff_hz >= DEFAULT_MAXIMUM_SPECTRUM_HZ:
            raise ValueError(f"shake cutoff must be below {DEFAULT_MAXIMUM_SPECTRUM_HZ:g} Hz")
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
        if not math.isfinite(self.gripper_period_s) or self.gripper_period_s <= 0.0:
            raise ValueError("gripper period must be positive and finite")
        if (
            not math.isfinite(self.gripper_limit_margin_fraction)
            or not 0.0 <= self.gripper_limit_margin_fraction < MAX_GRIPPER_LIMIT_MARGIN_FRACTION
        ):
            raise ValueError("gripper limit margin must be finite and in [0, 0.5)")
        if not self.move_gripper and self.gripper_min_rad is not None:
            raise ValueError("gripper position bounds cannot be used with hold gripper")

    @property
    def transition_duration_s(self) -> float:
        """Acceleration/deceleration time at each end of a trajectory setting."""
        return self.period_s * FIGURE_EIGHT_TRANSITION_PERIOD_FRACTION

    @property
    def setting_duration_s(self) -> float:
        """Duration of two loops plus the time needed to start and stop."""
        return FIGURE_EIGHT_LOOPS_PER_SETTING * self.period_s + self.transition_duration_s

    @property
    def duration_s(self) -> float:
        return self.setting_duration_s * len(FIGURE_EIGHT_SETTINGS)

    @property
    def gripper_cycle_count(self) -> int:
        """Fit at least one whole gripper cycle near the requested average rate."""
        return max(1, math.floor(self.duration_s / self.gripper_period_s + BOUNDS_EPSILON))

    @property
    def effective_gripper_period_s(self) -> float:
        """Stretch whole gripper cycles to fill the figure-eight duration."""
        return self.duration_s / self.gripper_cycle_count


@dataclass(frozen=True, kw_only=True)
class TrackingSample:
    """One simultaneous desired-versus-measured controller sample."""

    segment: Segment
    setting: str
    elapsed_s: float
    state_timestamp_s: float
    reference_position_m: NDArray[np.float64]
    osc_position_m: NDArray[np.float64]
    measured_position_m: NDArray[np.float64]
    osc_position_error_m: NDArray[np.float64]
    end_to_end_position_error_m: NDArray[np.float64]
    osc_orientation_error_rad: float
    end_to_end_orientation_error_rad: float
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
    commanded_linear_velocity_m_s: NDArray[np.float64] | None = None
    commanded_angular_velocity_rad_s: NDArray[np.float64] | None = None


@dataclass(frozen=True, kw_only=True)
class ErrorStatistics:
    rms: float
    mean: float
    p95: float
    maximum: float


StatisticsTableRow = tuple[str, ErrorStatistics, float]


@dataclass(frozen=True, kw_only=True)
class TrackingStatistics:
    osc_position_m: ErrorStatistics
    osc_orientation_rad: ErrorStatistics
    end_to_end_position_m: ErrorStatistics
    end_to_end_orientation_rad: ErrorStatistics
    arm_joint_position_rad: tuple[ErrorStatistics, ...]
    gripper_position_rad: tuple[ErrorStatistics, ...]


@dataclass(frozen=True, kw_only=True)
class TrackingRun:
    samples: list[TrackingSample]
    completed: bool
    failure_reason: str | None = None
    controller_command_timings: list[ControllerCommandTiming] = field(default_factory=list)
    native_joint_samples: list[NativeJointSample] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class RuntimeFeedback:
    """Latest controller output and hardware feedback with receipt times."""

    state: RobotState
    joint_command: RobotJointCommand
    mode: Mode
    last_state_received_s: float
    last_joint_command_received_s: float
    last_mode_received_s: float
