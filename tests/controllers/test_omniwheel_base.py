from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.controllers.omniwheel_base import OmniwheelBaseController
from humanoid.robots.base import Robot
from humanoid.types.controllers import OmniwheelBaseConfig
from humanoid.types.homing import HomingPreset


@pytest.fixture(scope="module")
def triskel_robot() -> Robot:
    return Robot(ROBOT_CONFIGS["triskel"])


@pytest.fixture
def controller(triskel_robot) -> OmniwheelBaseController:
    return OmniwheelBaseController(triskel_robot)


def test_uses_default_config(triskel_robot):
    controller = OmniwheelBaseController(triskel_robot)
    assert isinstance(controller.config, OmniwheelBaseConfig)


def test_requires_mobile_base(triskel_robot):
    config = replace(
        triskel_robot.config,
        base=None,
        state_estimation=None,
        omniwheel_base_config=None,
    )
    robot = Robot.__new__(Robot)
    robot.__dict__.update(triskel_robot.__dict__)
    robot._config = config

    with pytest.raises(ValueError, match="requires a configured planar base"):
        OmniwheelBaseController(robot)


def test_owns_only_root_and_wheels(controller, triskel_robot):
    names = [
        triskel_robot.joint_idx_to_name(index) for index in controller.controlled_joint_indices
    ]
    assert names == ["root_joint", "wheel_1", "wheel_2", "wheel_3"]


def test_requires_state_before_control(controller):
    with pytest.raises(RuntimeError, match="not initialized"):
        controller.compute_control(pin.SE3.Identity())


def test_moves_base_and_wheels_without_moving_arm_or_gripper(triskel_robot):
    controller = OmniwheelBaseController(triskel_robot)
    q = triskel_robot.config.homing_presets[HomingPreset.HOME].copy()
    controller.update_state(q)
    target = triskel_robot.get_base_pose(q)
    assert target is not None
    target.translation[0] += 1.0

    result = controller.compute_control(target)

    arm_joint_indices = triskel_robot.get_arm_joint_indices()
    arm_q_indices = triskel_robot.get_joint_position_indices(arm_joint_indices)
    arm_v_indices = triskel_robot.get_joint_velocity_indices(arm_joint_indices)
    gripper_q_indices = triskel_robot.get_gripper_position_indices()
    gripper_v_indices = triskel_robot.get_joint_velocity_indices(
        triskel_robot.config.gripper_joint_indices or []
    )
    root_v_slice = triskel_robot.get_root_v_slice()
    base = triskel_robot.config.base
    assert root_v_slice is not None
    assert base is not None
    np.testing.assert_allclose(result.q[arm_q_indices], q[arm_q_indices])
    np.testing.assert_allclose(result.q[gripper_q_indices], q[gripper_q_indices])
    np.testing.assert_array_equal(result.v[arm_v_indices], 0.0)
    np.testing.assert_array_equal(result.v[gripper_v_indices], 0.0)
    motion_tolerance = 1e-8
    assert np.any(np.abs(result.v[controller.controlled_v_indices]) > motion_tolerance)
    assert abs(result.v[root_v_slice][0]) <= base.velocity_limits.linear + 1e-9
    assert abs(result.v[root_v_slice][1]) <= base.velocity_limits.linear + 1e-9
    assert abs(result.v[root_v_slice][2]) <= base.velocity_limits.angular + 1e-9
    np.testing.assert_allclose(result.v[root_v_slice], [0.2, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(
        result.v[controller.controlled_v_indices[3:]],
        [4.0, -2.0, -2.0],
        atol=3e-6,
    )


def test_updates_base_target(controller, triskel_robot):
    q = triskel_robot.config.homing_presets[HomingPreset.HOME]
    controller.update_state(q)
    target = pin.SE3(np.eye(3), np.array([1.0, 0.5, 0.0]))

    controller.compute_control(target)

    assert controller.target_pose is not None
    np.testing.assert_allclose(controller.target_pose.translation, target.translation, atol=1e-9)


def test_scales_motion_to_respect_wheel_velocity_limits(triskel_robot):
    controller = OmniwheelBaseController(triskel_robot)
    q = triskel_robot.config.homing_presets[HomingPreset.HOME].copy()
    controller.update_state(q)
    target = triskel_robot.get_base_pose(q)
    assert target is not None
    target.translation[:2] += 1.0
    target.rotation = pin.utils.rotate("z", 1.0) @ target.rotation

    result = controller.compute_control(target)

    wheel_v_indices = triskel_robot.get_joint_velocity_indices(
        triskel_robot.get_wheel_joint_indices()
    )
    np.testing.assert_array_less(
        np.abs(result.v[wheel_v_indices]),
        triskel_robot.model.velocityLimit[wheel_v_indices] + 1e-12,
    )


def test_small_pose_error_is_corrected_in_one_tick(triskel_robot):
    config = OmniwheelBaseConfig(dt=0.01)
    controller = OmniwheelBaseController(triskel_robot, config)
    q = triskel_robot.config.homing_presets[HomingPreset.HOME].copy()
    controller.update_state(q)
    target = triskel_robot.get_base_pose(q)
    assert target is not None
    target.translation[0] += 0.001

    result = controller.compute_control(target)

    root_v_slice = triskel_robot.get_root_v_slice()
    assert root_v_slice is not None
    np.testing.assert_allclose(result.v[root_v_slice], [0.1, 0.0, 0.0], atol=1e-9)


def test_runtime_timestep_overrides_configured_timestep(triskel_robot):
    controller = OmniwheelBaseController(triskel_robot, OmniwheelBaseConfig(dt=0.1))
    q = triskel_robot.config.homing_presets[HomingPreset.HOME].copy()
    controller.update_state(q)
    target = triskel_robot.get_base_pose(q)
    assert target is not None
    target.translation[0] += 0.001

    result = controller.compute_control(target, dt=0.01)

    root_v_slice = triskel_robot.get_root_v_slice()
    assert root_v_slice is not None
    np.testing.assert_allclose(result.v[root_v_slice], [0.1, 0.0, 0.0], atol=1e-9)
