"""Controller configuration, outputs, and tracking diagnostics."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from humanoid.types.actuator import ActuatorEffortSource, ActuatorEffortUnit


class TaskName(StrEnum):
    """Task names used in the operational-space controller."""

    TOOL = "tool"
    MANIPULABILITY = "manipulability"
    DAMPING = "damping"
    LOW_ACCELERATION = "low_acceleration"


@dataclass
class ControlResult:
    """Full-model position and velocity output from one controller."""

    q: NDArray[np.float64]
    v: NDArray[np.float64]


@dataclass
class OperationalSpaceConfig:
    """Configuration parameters for arm operational-space control.

    Args:
        tool_position_cost: Weight on the tool-frame position tracking task
            in the IK QP. Higher values track the commanded tool position
            more aggressively at the expense of other tasks.
        tool_orientation_cost: Weight on the tool-frame orientation tracking
            task in the IK QP.
        dt: Controller integration timestep in seconds; also sets the rate
            at which this controller is expected to be ticked.
        manipulability_cost: Scalar weight on the soft objective that increases
            the regularized log-volume of the arm's tool Jacobian. This favors
            Cartesian dexterity and competes with tool tracking and damping.
            Zero disables the task.
        manipulability_regularization: Positive diagonal added to the arm
            Jacobian Gram matrix before evaluating log manipulability. Keeps
            the measure and gradient finite near singular configurations.
        damping_cost: Scalar weight on the velocity-damping regularizer that
            penalizes joint velocities. Multiplied element-wise with
            ``damping_mask``.
        low_acceleration_cost: Scalar weight on the change in joint velocity
            between controller ticks. Multiplied element-wise with
            ``low_acceleration_mask``. Zero disables the soft smoothing task.
        joint_velocity_limit: Optional maximum arm-joint velocity in rad/s.
            A scalar applies to every arm joint and an array specifies one
            value per arm velocity. This is enforced in addition to the URDF.
        joint_acceleration_limit: Maximum arm-joint acceleration in rad/s^2.
            A scalar applies to every arm joint and an array specifies one
            value per arm velocity. ``None`` disables the hard limit.
        joint_position_margin: Distance inside joint position bounds reserved by
            the braking constraint (radians for revolute joints). Requires a
            joint acceleration limit. This bounds commands, not physical overshoot.
        solver: Name of the QP solver backend passed to Pink (e.g.
            ``"quadprog"``).
        avoid_collisions: Whether to enable self-collision avoidance
            barriers using the robot's collision model and SRDF pairs.
        min_collision_distance: Minimum distance between any collision pairs.
            Defaults to 0.02.
        collision_safe_displacement_gain: Weight on the self-collision barrier's
            safe-displacement regularizer. This regularizer favors zero joint
            displacement but does not set the hard minimum-distance constraint.
        damping_mask: Per-joint multiplier on ``damping_cost``. A scalar applies
            to all arm joints; an array specifies one value per arm velocity or
            per non-root velocity. Non-arm coordinates are always excluded.
        low_acceleration_mask: Per-joint multiplier on
            ``low_acceleration_cost``, with the same scalar/array semantics as
            ``damping_mask``.
    """

    tool_position_cost: float = 1.0
    tool_orientation_cost: float = 1.0
    dt: float = 0.005
    manipulability_cost: float = 1e-3
    manipulability_regularization: float = 1e-6
    damping_cost: float = 1e-1
    low_acceleration_cost: float = 0.0
    joint_velocity_limit: np.ndarray | float | None = None
    joint_acceleration_limit: np.ndarray | float | None = None
    joint_position_margin: float = 0.0
    solver: str = "quadprog"
    avoid_collisions: bool = False
    min_collision_distance: float = 0.02
    collision_safe_displacement_gain: float = 1.0
    damping_mask: np.ndarray | float = 1.0
    low_acceleration_mask: np.ndarray | float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Operational-space timestep must be positive and finite.")
        if not np.isfinite(self.manipulability_cost) or self.manipulability_cost < 0.0:
            raise ValueError("Manipulability cost must be finite and non-negative.")
        if (
            not np.isfinite(self.manipulability_regularization)
            or self.manipulability_regularization <= 0.0
        ):
            raise ValueError("Manipulability regularization must be positive and finite.")
        if not np.isfinite(self.low_acceleration_cost) or self.low_acceleration_cost < 0.0:
            raise ValueError("Low-acceleration cost must be finite and non-negative.")
        if not np.isfinite(self.joint_position_margin) or self.joint_position_margin < 0.0:
            raise ValueError("Joint position margin must be finite and non-negative.")
        if self.joint_position_margin > 0.0 and self.joint_acceleration_limit is None:
            raise ValueError("Joint position margin requires a joint acceleration limit.")
        if (
            not np.isfinite(self.collision_safe_displacement_gain)
            or self.collision_safe_displacement_gain < 0.0
        ):
            raise ValueError("Collision safe-displacement gain must be finite and non-negative.")
        for name, configured_limit in (
            ("velocity", self.joint_velocity_limit),
            ("acceleration", self.joint_acceleration_limit),
        ):
            if configured_limit is None:
                continue
            limit = np.asarray(configured_limit, dtype=float)
            if not np.isfinite(limit).all() or np.any(limit <= 0.0):
                raise ValueError(f"Joint {name} limits must be positive and finite.")


@dataclass
class OmniwheelBaseConfig:
    """Configuration parameters for analytic omniwheel pose control.

    The dimensionless gains scale the pose error corrected on each controller
    tick. A gain of one requests deadbeat correction before velocity limiting.
    """

    position_gain: float = 1.0
    orientation_gain: float = 1.0
    dt: float = 0.005

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Omniwheel-base timestep must be positive and finite.")


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
