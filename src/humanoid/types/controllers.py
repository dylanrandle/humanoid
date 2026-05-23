from dataclasses import dataclass

import numpy as np


@dataclass
class OperationalSpaceConfig:
    """Configuration parameters for the operational space controller.

    Args:
        tool_position_cost: Weight on the tool-frame position tracking task
            in the IK QP. Higher values track the commanded tool position
            more aggressively at the expense of other tasks.
        tool_orientation_cost: Weight on the tool-frame orientation tracking
            task in the IK QP.
        base_position_cost: Weight on the base-frame position tracking task
            (when ``base_frame`` is configured on the robot).
        base_orientation_cost: Weight on the base-frame orientation tracking
            task (when ``base_frame`` is configured on the robot).
        wheel_cost: Weight on each wheel's rolling/omniwheel constraint task
            that enforces no-slip motion between the wheel frame and floor.
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
        joint_centering_mask: Per-joint multiplier on
            ``joint_centering_cost``. A scalar applies the same weight to
            all joints; an array selectively centers some joints (e.g. arm)
            while leaving others (e.g. base) free.
        damping_mask: Per-joint multiplier on ``damping_cost``, with the
            same scalar/array semantics as ``joint_centering_mask``.
    """

    tool_position_cost: float = 1.0
    tool_orientation_cost: float = 1.0
    base_position_cost: float = 1.0
    base_orientation_cost: float = 1.0
    wheel_cost: float = 10.0
    dt: float = 0.005
    joint_centering_cost: float = 1e-3
    damping_cost: float = 1e-1
    solver: str = "quadprog"
    avoid_collisions: bool = False
    joint_centering_mask: np.ndarray | float = 1.0
    damping_mask: np.ndarray | float = 1.0
