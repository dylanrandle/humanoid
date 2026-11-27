import json

import numpy as np
import pytest

from humanoid.motion.validation import motion_target
from humanoid.types.motion import ControlOperation
from humanoid.types.trial import TrialStep


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
        motion_target(operation, arguments)
    with pytest.raises(ValueError):
        TrialStep.model_validate_json(
            json.dumps({"command": {"operation": operation, "arguments": arguments}, "memory": {}})
        )


def test_wire_commands_translate_to_typed_targets():
    base = motion_target(ControlOperation.MOVE_BASE, {"delta": [0.04, 0, 0], "duration": 0.2})
    assert base.base_delta is not None
    np.testing.assert_allclose(base.base_delta, [0.04, 0, 0])
    assert base.duration == pytest.approx(0.2)
    tool = motion_target(
        ControlOperation.MOVE_TOOL, {"position": [0, 0, 0.2], "quaternion": [0, 0, 0, 1]}
    )
    assert tool.tool_pose is not None
    np.testing.assert_allclose(tool.tool_pose.translation, [0, 0, 0.2])
    np.testing.assert_allclose(tool.tool_pose.rotation, np.eye(3))
    gripper = motion_target(ControlOperation.SET_GRIPPER, {"positions": [1.2]})
    assert gripper.gripper_positions is not None
    np.testing.assert_allclose(gripper.gripper_positions, [1.2])


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        (ControlOperation.HOLD, {}),
        (ControlOperation.MOVE_BASE, {"delta": [0.1, 0, 0]}),
        (ControlOperation.MOVE_TOOL, {"position": [0, 0, 0.2], "quaternion": [0, 0, 0, 1]}),
        (ControlOperation.SET_GRIPPER, {"positions": [0.5]}),
    ],
)
def test_api_and_trial_share_motion_defaults_and_validation(operation, arguments):
    target = motion_target(operation, arguments)
    step = TrialStep.model_validate_json(
        json.dumps({"command": {"operation": operation, "arguments": arguments}, "memory": {}})
    )
    assert step.command is not None
    assert step.command.arguments.duration == target.duration == 1.0
    normalized = motion_target(operation, step.command.arguments.model_dump(mode="json"))
    assert normalized.duration == target.duration
