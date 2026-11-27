"""Validate shared motion arguments and translate them into native policy targets."""

from collections.abc import Mapping
from typing import assert_never

import numpy as np
import pinocchio as pin

from humanoid.types.motion import (
    ControlOperation,
    Hold,
    MotionTarget,
    MoveBase,
    MoveTool,
    SetGripper,
)


def motion_target(operation: ControlOperation, arguments: Mapping[str, object]) -> MotionTarget:
    if operation is ControlOperation.MOVE_TOOL:
        tool = MoveTool.model_validate(arguments)
        return MotionTarget(
            duration=tool.duration,
            tool_pose=pin.SE3(
                pin.Quaternion(np.asarray(tool.quaternion)).matrix(), np.asarray(tool.position)
            ),
        )
    if operation is ControlOperation.MOVE_BASE:
        base = MoveBase.model_validate(arguments)
        return MotionTarget(duration=base.duration, base_delta=np.asarray(base.delta))
    if operation is ControlOperation.SET_GRIPPER:
        gripper = SetGripper.model_validate(arguments)
        return MotionTarget(
            duration=gripper.duration, gripper_positions=np.asarray(gripper.positions)
        )
    if operation is ControlOperation.HOLD:
        hold = Hold.model_validate(arguments)
        return MotionTarget(duration=hold.duration)
    assert_never(operation)
