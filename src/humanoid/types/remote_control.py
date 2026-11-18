"""Commands, outcomes, and configuration for the remote control policy."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pinocchio as pin


@dataclass(frozen=True)
class PendingCommand:
    command: MotionCommand
    result: Future[MotionResult]
    deadline: float | None = None


@dataclass(frozen=True, kw_only=True)
class MotionCommand:
    """Optional tool/base/gripper targets over a bounded duration.

    An empty command holds the current pose. Tool poses use the robot's tool
    command frame; base_delta is [x, y, yaw] relative to the base at execution
    start. Omitted gripper targets preserve the existing squeeze target.
    """

    duration: float = 1.0
    tool_pose: pin.SE3 | None = None
    base_delta: NDArray[np.float64] | None = None
    gripper_positions: NDArray[np.float64] | None = None


class MotionResult(TypedDict):
    """Measured motion outcome; completion does not imply reaching the target."""

    reached: bool
    position_error_m: float
    orientation_error_rad: float
    gripper_error_rad: float
    elapsed_seconds: float
    step_id: int


@dataclass
class RemoteMotion:
    duration: float
    started_at: float
    initial: pin.SE3
    target: pin.SE3
    base_target: pin.SE3
    initial_gripper: NDArray[np.float64]
    gripper: NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class RemoteControlPolicyConfig:
    """Policy timing, bounded command horizons, and feedback requirements.

    Tool/base speeds come from RobotConfig, and joint bounds come from the URDF.
    Gripper closing time sets the rate of traversing that joint range, as in teleop.
    The workspace bounds constrain the tool origin, not collision clearance.
    """

    dt: float = 0.02
    feedback_timeout: float = 0.25
    max_duration: float = 2.0
    max_tool_translation: float = 0.08
    max_tool_rotation: float = 0.5
    max_base_translation: float = 0.15
    max_base_rotation: float = 0.5
    gripper_close_time: float = 3.0
    tool_workspace_min: tuple[float, float, float] = (-0.35, -0.35, 0.01)
    tool_workspace_max: tuple[float, float, float] = (0.35, 0.45, 0.48)
    position_tolerance: float = 0.008
    orientation_tolerance: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "dt",
            "feedback_timeout",
            "max_duration",
            "max_tool_translation",
            "max_tool_rotation",
            "max_base_translation",
            "max_base_rotation",
            "gripper_close_time",
            "position_tolerance",
            "orientation_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for value in (self.tool_workspace_min, self.tool_workspace_max):
            if np.shape(value) != (3,) or not np.isfinite(value).all():
                raise ValueError("Tool workspace must use finite 3D coordinates.")
        if np.any(np.asarray(self.tool_workspace_min) >= self.tool_workspace_max):
            raise ValueError("Tool workspace minimum must be below maximum.")


class CommandExpiredError(RuntimeError):
    """The command was not started before its receiver-local deadline."""


class CommandCancelledError(RuntimeError):
    """An operator action or caller interrupted the command."""
