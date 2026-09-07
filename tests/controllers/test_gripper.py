from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.controllers.gripper import GripperController
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset


@pytest.fixture(scope="module")
def panda_robot() -> Robot:
    return Robot(ROBOT_CONFIGS["panda"])


@pytest.fixture(scope="module")
def triskel_robot() -> Robot:
    return Robot(ROBOT_CONFIGS["triskel"])


def test_requires_configured_gripper(panda_robot):
    config = replace(panda_robot.config, gripper_joint_indices=None)
    robot = Robot.__new__(Robot)
    robot.__dict__.update(panda_robot.__dict__)
    robot._config = config

    with pytest.raises(ValueError, match="requires configured gripper"):
        GripperController(robot)


def test_requires_state_before_control(panda_robot):
    controller = GripperController(panda_robot)

    with pytest.raises(RuntimeError, match="not initialized"):
        controller.compute_control(np.array([0.01]))


def test_controls_only_gripper_position(panda_robot):
    controller = GripperController(panda_robot)
    q = panda_robot.config.homing_presets[HomingPreset.HOME].copy()
    target = np.array([0.0123])
    controller.update_state(q)

    result = controller.compute_control(target)

    gripper_indices = panda_robot.get_gripper_position_indices()
    arm_indices = panda_robot.get_joint_position_indices(panda_robot.get_arm_joint_indices())
    np.testing.assert_allclose(result.q[gripper_indices], target)
    np.testing.assert_allclose(result.q[arm_indices], q[arm_indices])
    np.testing.assert_array_equal(result.v, np.zeros(panda_robot.model.nv))


def test_mobile_gripper_uses_position_index_not_joint_index(triskel_robot):
    controller = GripperController(triskel_robot)
    q = triskel_robot.config.homing_presets[HomingPreset.HOME].copy()
    joint_idx = triskel_robot.config.gripper_joint_indices[0]
    position_idx = triskel_robot.joint_idx_to_position_idx(joint_idx)
    target = np.array([0.123])
    assert joint_idx != position_idx
    controller.update_state(q)

    result = controller.compute_control(target)

    assert result.q[position_idx] == pytest.approx(target[0])
    assert result.q[joint_idx] == pytest.approx(q[joint_idx])


@pytest.mark.parametrize(("target", "limit_index"), [(-1.0, 0), (99.0, 1)])
def test_clamps_target_to_joint_limits(panda_robot, target, limit_index):
    controller = GripperController(panda_robot)
    controller.update_state(pin.neutral(panda_robot.model))

    result = controller.compute_control(np.array([target]))

    gripper_index = panda_robot.get_gripper_position_indices()[0]
    expected = panda_robot.get_gripper_limits()[0][limit_index]
    assert result.q[gripper_index] == pytest.approx(expected)


@pytest.mark.parametrize("target", [np.array([]), np.array([0.1, 0.2])])
def test_rejects_wrong_target_shape(panda_robot, target):
    controller = GripperController(panda_robot)
    controller.update_state(pin.neutral(panda_robot.model))

    with pytest.raises(ValueError, match="target must have shape"):
        controller.compute_control(target)
