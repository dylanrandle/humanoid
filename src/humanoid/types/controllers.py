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
        solver: Name of the QP solver backend passed to Pink (e.g.
            ``"quadprog"``).
        avoid_collisions: Whether to enable self-collision avoidance
            barriers using the robot's collision model and SRDF pairs.
        min_collision_distance: Minimum distance between any collision pairs.
            Defaults to 0.02.
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
    solver: str = "quadprog"
    avoid_collisions: bool = False
    min_collision_distance: float = 0.02
    joint_centering_mask: np.ndarray | float = 1.0
    damping_mask: np.ndarray | float = 1.0


@dataclass
class OmniwheelBaseConfig:
    """Configuration parameters for analytic omniwheel pose control.

    The dimensionless gains scale the pose error corrected on each controller
    tick. A gain of one requests deadbeat correction before velocity limiting.
    """

    position_gain: float = 1.0
    orientation_gain: float = 1.0
    dt: float = 0.005
