import numpy as np
import pytest

from humanoid.remote_control.commands import motion_command
from humanoid.types.remote_api import ControlOperation


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        (ControlOperation.MOVE_BASE, {"delta": ["0", 0, 0]}),
        (ControlOperation.MOVE_BASE, {"delta": [True, 0, 0]}),
        (ControlOperation.MOVE_BASE, {"delta": [float("nan"), 0, 0]}),
        (ControlOperation.MOVE_BASE, {"delta": [0, 0]}),
        (ControlOperation.MOVE_TOOL, {"position": [0, 0, 0.2], "quaternion": [0, 0, 0, 0]}),
        (ControlOperation.HOLD, {"duration": True}),
        (ControlOperation.HOLD, {"duration": "1"}),
        (ControlOperation.HOLD, {"extra": 1}),
    ],
)
def test_malformed_wire_commands_are_rejected(operation, arguments):
    with pytest.raises(ValueError):
        motion_command(operation, arguments)


def test_wire_commands_translate_to_typed_targets():
    base = motion_command(ControlOperation.MOVE_BASE, {"delta": [0.04, 0, 0], "duration": 0.2})
    assert base.base_delta is not None
    np.testing.assert_allclose(base.base_delta, [0.04, 0, 0])
    assert base.duration == pytest.approx(0.2)
    tool = motion_command(
        ControlOperation.MOVE_TOOL, {"position": [0, 0, 0.2], "quaternion": [0, 0, 0, 1]}
    )
    assert tool.tool_pose is not None
    np.testing.assert_allclose(tool.tool_pose.translation, [0, 0, 0.2])
    np.testing.assert_allclose(tool.tool_pose.rotation, np.eye(3))
    gripper = motion_command(ControlOperation.SET_GRIPPER, {"positions": [1.2]})
    assert gripper.gripper_positions is not None
    np.testing.assert_allclose(gripper.gripper_positions, [1.2])
