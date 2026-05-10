import numpy as np
import pinocchio as pin
import pytest

from humanoid.types.action import Action


def test_tool_action_with_base_pose():
    """Tool-space action can carry an optional base_pose."""
    tool_pose = pin.SE3.Identity()
    base_pose = pin.SE3(pin.utils.rotate("z", 0.5), np.array([1.0, 2.0, 0.0]))

    action = Action(tool_pose=tool_pose, base_pose=base_pose)

    assert action.is_tool_action
    assert not action.is_joint_action
    assert action.base_pose is base_pose


def test_joint_action_with_base_pose():
    """Joint-space action can also carry an optional base_pose."""
    joint_positions = np.zeros(7)
    base_pose = pin.SE3.Identity()

    action = Action(joint_positions=joint_positions, base_pose=base_pose)

    assert action.is_joint_action
    assert not action.is_tool_action
    assert action.base_pose is base_pose


def test_action_base_pose_defaults_to_none():
    """base_pose defaults to None when not provided."""
    action = Action(tool_pose=pin.SE3.Identity())
    assert action.base_pose is None


def test_action_validation_still_requires_one_of_joint_or_tool():
    """Specifying only base_pose without joint_positions or tool_pose raises."""
    with pytest.raises(ValueError, match="Either joint_positions or tool_pose"):
        Action(base_pose=pin.SE3.Identity())


def test_action_validation_rejects_both_joint_and_tool():
    """Specifying both joint_positions and tool_pose still raises, even with base_pose."""
    with pytest.raises(ValueError, match="Only one of joint_positions or tool_pose"):
        Action(
            joint_positions=np.zeros(7),
            tool_pose=pin.SE3.Identity(),
            base_pose=pin.SE3.Identity(),
        )
