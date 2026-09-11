from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
        joint_centering_cost: Scalar weight on the posture (joint-centering)
            null-space task that pulls joints toward the home position.
            Multiplied element-wise with ``joint_centering_mask``.
        damping_cost: Scalar weight on the velocity-damping regularizer that
            penalizes joint velocities. Multiplied element-wise with
            ``damping_mask``.
        low_acceleration_cost: Scalar weight on the change in joint velocity
            between controller ticks. Zero disables the soft smoothing task.
        joint_velocity_limit: Optional maximum arm-joint velocity in rad/s.
            A scalar applies to every arm joint and an array specifies one
            value per arm velocity. This is enforced in addition to the URDF.
        joint_acceleration_limit: Maximum arm-joint acceleration in rad/s^2.
            A scalar applies to every arm joint and an array specifies one
            value per arm velocity. ``None`` disables the hard limit.
        solver: Name of the QP solver backend passed to Pink (e.g.
            ``"quadprog"``).
        avoid_collisions: Whether to enable self-collision avoidance
            barriers using the robot's collision model and SRDF pairs.
        min_collision_distance: Minimum distance between any collision pairs.
            Defaults to 0.02.
        collision_safe_displacement_gain: Weight on the self-collision barrier's
            safe-displacement regularizer. This regularizer favors zero joint
            displacement but does not set the hard minimum-distance constraint.
        joint_centering_mask: Per-joint multiplier on
            ``joint_centering_cost``. A scalar applies the same weight to
            all arm joints; an array can selectively center individual joints.
            Non-arm coordinates are always excluded by the controller.
        damping_mask: Per-joint multiplier on ``damping_cost``, with the
            same scalar/array semantics as ``joint_centering_mask``.
    """

    tool_position_cost: float = 1.0
    tool_orientation_cost: float = 1.0
    dt: float = 0.005
    joint_centering_cost: float = 1e-3
    damping_cost: float = 1e-1
    low_acceleration_cost: float = 0.0
    joint_velocity_limit: np.ndarray | float | None = None
    joint_acceleration_limit: np.ndarray | float | None = None
    solver: str = "quadprog"
    avoid_collisions: bool = False
    min_collision_distance: float = 0.02
    collision_safe_displacement_gain: float = 1.0
    joint_centering_mask: np.ndarray | float = 1.0
    damping_mask: np.ndarray | float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Operational-space timestep must be positive and finite.")
        if not np.isfinite(self.low_acceleration_cost) or self.low_acceleration_cost < 0.0:
            raise ValueError("Low-acceleration cost must be finite and non-negative.")
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
