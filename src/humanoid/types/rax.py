"""RAX tool names and outcomes; transport is supplied by the general robot API."""

from enum import StrEnum

from humanoid.types.remote_control import MotionResult


class RaxOperation(StrEnum):
    """Named autonomy motion and cancellation operations."""

    MOVE_TOOL = "move_tool"
    MOVE_BASE = "move_base"
    GRIPPER = "gripper"
    HOLD = "hold"
    CANCEL = "cancel"


class RaxCommandResult(MotionResult):
    operation: RaxOperation
