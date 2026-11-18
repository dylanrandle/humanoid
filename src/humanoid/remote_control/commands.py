"""Translate remote API arguments into typed robot motion commands."""

import math
from collections.abc import Mapping

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.types.remote_api import ControlOperation
from humanoid.types.remote_control import MotionCommand


def motion_command(operation: ControlOperation, arguments: Mapping[str, object]) -> MotionCommand:
    duration = arguments.get("duration", 1.0)
    if (type(duration) is not int and type(duration) is not float) or not math.isfinite(duration):
        raise ValueError("duration must be a finite number.")
    if operation is ControlOperation.MOVE_TOOL:
        _keys(arguments, {"position", "quaternion", "duration"}, {"position", "quaternion"})
        position = _vector(arguments["position"], 3)
        quaternion = _vector(arguments["quaternion"], 4)
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-5):
            raise ValueError("quaternion must have unit norm, in x,y,z,w order.")
        return MotionCommand(
            duration=duration, tool_pose=pin.SE3(pin.Quaternion(quaternion).matrix(), position)
        )
    if operation is ControlOperation.MOVE_BASE:
        _keys(arguments, {"delta", "duration"}, {"delta"})
        return MotionCommand(duration=duration, base_delta=_vector(arguments["delta"], 3))
    if operation is ControlOperation.SET_GRIPPER:
        _keys(arguments, {"positions", "duration"}, {"positions"})
        positions = arguments["positions"]
        if not isinstance(positions, (list, tuple)):
            raise ValueError("Gripper positions must be a numeric vector.")
        return MotionCommand(
            duration=duration, gripper_positions=_vector(positions, len(positions))
        )
    if operation is ControlOperation.HOLD:
        _keys(arguments, {"duration"}, set())
        return MotionCommand(duration=duration)
    raise ValueError(f"Unsupported motion operation {operation!r}.")


def _keys(arguments: Mapping[str, object], allowed: set[str], required: set[str]) -> None:
    if arguments.keys() - allowed or required - arguments.keys():
        raise ValueError(
            f"Expected keys {sorted(required)} and optional {sorted(allowed - required)}."
        )


def _vector(value: object, length: int) -> NDArray[np.float64]:
    if not isinstance(value, (list, tuple)) or any(type(x) not in (int, float) for x in value):
        raise ValueError("Vector entries must be numbers.")
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected a finite vector of length {length}.")
    return vector
