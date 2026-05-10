import numpy as np
import pinocchio as pin

from humanoid.types.action import Action


def test_tool_action_with_base_pose():
    tool_pose = pin.SE3.Identity()
    base_pose = pin.SE3(pin.utils.rotate("z", 0.5), np.array([1.0, 2.0, 0.0]))
    action = Action(tool_pose=tool_pose, base_pose=base_pose)
    assert action.tool_pose is tool_pose
    assert action.base_pose is base_pose
    assert action.joint_positions is None


def test_joint_action_with_base_pose():
    joint_positions = np.zeros(7)
    base_pose = pin.SE3.Identity()
    action = Action(joint_positions=joint_positions, base_pose=base_pose)
    assert action.joint_positions is joint_positions
    assert action.base_pose is base_pose
    assert action.tool_pose is None


def test_action_base_pose_defaults_to_none():
    action = Action(tool_pose=pin.SE3.Identity())
    assert action.base_pose is None


def test_action_allows_both_joint_and_tool():
    action = Action(joint_positions=np.zeros(7), tool_pose=pin.SE3.Identity())
    assert action.joint_positions is not None
    assert action.tool_pose is not None


def test_action_allows_all_none():
    action = Action()
    assert action.joint_positions is None
    assert action.tool_pose is None
    assert action.base_pose is None
    assert action.gripper_positions is None
