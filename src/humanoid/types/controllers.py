from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


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
