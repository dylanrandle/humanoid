from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest
from pink.limits import FloatingBaseVelocityLimit
from pink.tasks import FrameTask, RelativeFrameTask

from humanoid.config import ROBOT_CONFIGS
from humanoid.controllers.operational_space import (
    ControlResult,
    OperationalSpaceConfig,
    OperationalSpaceController,
    TaskName,
)
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset


@pytest.fixture(scope="module")
def panda_robot() -> Robot:
    """Fixed-base 7-DOF Panda arm — no base frame configured."""
    return Robot(ROBOT_CONFIGS["panda"])


@pytest.fixture(scope="module")
def mobile_robot() -> Robot:
    """Mobile manipulator with a base frame configured."""
    return Robot(ROBOT_CONFIGS["elrobot_mobile"])


@pytest.fixture
def panda_osc(panda_robot) -> OperationalSpaceController:
    return OperationalSpaceController(robot=panda_robot)


@pytest.fixture
def mobile_osc(mobile_robot) -> OperationalSpaceController:
    return OperationalSpaceController(robot=mobile_robot)


class TestConstruction:
    def test_default_config_used_when_none(self, panda_robot):
        osc = OperationalSpaceController(robot=panda_robot, config=None)
        assert isinstance(osc.config, OperationalSpaceConfig)

    def test_explicit_config_kept(self, panda_robot):
        config = OperationalSpaceConfig(dt=0.05, tool_position_cost=42.0)
        osc = OperationalSpaceController(robot=panda_robot, config=config)
        assert osc.config is config

    def test_tasks_for_fixed_base_robot(self, panda_osc):
        """A fixed robot gets a world-frame tool task and no base task."""
        assert isinstance(panda_osc.tasks[TaskName.TOOL], FrameTask)
        assert TaskName.TOOL in panda_osc.tasks
        assert TaskName.JOINT_CENTERING in panda_osc.tasks
        assert TaskName.DAMPING in panda_osc.tasks
        assert TaskName.BASE not in panda_osc.tasks

    def test_tasks_for_mobile_robot_include_base(self, mobile_osc):
        """A mobile robot tracks its tool relative to its base."""
        assert isinstance(mobile_osc.tasks[TaskName.TOOL], RelativeFrameTask)
        assert TaskName.BASE in mobile_osc.tasks

    def test_mobile_robot_configures_floating_base_velocity_limit(self, mobile_osc):
        limit = mobile_osc.robot.model.floating_base_velocity_limit
        base_config = mobile_osc.robot.config.base

        assert isinstance(limit, FloatingBaseVelocityLimit)
        assert base_config is not None
        np.testing.assert_allclose(
            limit.linear_max,
            base_config.velocity_limits.linear,
        )
        np.testing.assert_allclose(
            limit.angular_max,
            base_config.velocity_limits.angular,
        )

    def test_invalid_tool_frame_raises(self, panda_robot):
        bad_tool_config = replace(panda_robot.config.tool, frame="not_a_real_frame")
        bad_config = replace(panda_robot.config, tool=bad_tool_config)
        bad_robot = Robot.__new__(Robot)
        bad_robot.__dict__.update(panda_robot.__dict__)
        bad_robot._config = bad_config
        with pytest.raises(ValueError, match="not found in URDF"):
            OperationalSpaceController(robot=bad_robot)

    def test_invalid_configured_base_frame_raises(self, mobile_robot):
        assert mobile_robot.config.base is not None
        bad_base_config = replace(mobile_robot.config.base, frame="not_a_real_frame")
        bad_config = replace(mobile_robot.config, base=bad_base_config)
        bad_robot = Robot.__new__(Robot)
        bad_robot.__dict__.update(mobile_robot.__dict__)
        bad_robot._config = bad_config
        with pytest.raises(ValueError, match="not found in URDF"):
            OperationalSpaceController(robot=bad_robot)

    def test_configuration_is_none_initially(self, panda_osc):
        """configuration is deferred until first update_state call."""
        assert panda_osc.configuration is None


class TestUpdateState:
    def test_first_call_initializes_configuration(self, panda_osc, panda_robot):
        q = panda_robot.config.homing_presets[HomingPreset.HOME]
        panda_osc.update_state(q)

        assert panda_osc.configuration is not None
        np.testing.assert_allclose(panda_osc.configuration.q, q)

    def test_subsequent_calls_update_configuration(self, panda_osc, panda_robot):
        q1 = panda_robot.config.homing_presets[HomingPreset.HOME]
        q2 = panda_robot.config.homing_presets[HomingPreset.REST]
        panda_osc.update_state(q1)
        configuration = panda_osc.configuration

        panda_osc.update_state(q2)

        # Same object reused, just updated.
        assert panda_osc.configuration is configuration
        np.testing.assert_allclose(panda_osc.configuration.q, q2)


class TestComputeControl:
    def test_raises_when_not_initialized(self, panda_osc):
        target = pin.SE3.Identity()
        with pytest.raises(RuntimeError, match="not initialized"):
            panda_osc.compute_control(target)

    def test_returns_control_result_with_expected_shapes(self, panda_osc, panda_robot):
        panda_osc.update_state(panda_robot.config.homing_presets[HomingPreset.HOME])
        target = pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.4]))

        result = panda_osc.compute_control(target)

        assert isinstance(result, ControlResult)
        assert result.q.shape == (panda_robot.model.nq,)
        assert result.v.shape == (panda_robot.model.nv,)

    def test_integrates_configuration_forward(self, panda_osc, panda_robot):
        """A reachable target should move the configuration toward it."""
        panda_osc.update_state(panda_robot.config.homing_presets[HomingPreset.HOME])
        q_before = panda_osc.configuration.q.copy()
        target = pin.SE3(np.eye(3), np.array([0.5, 0.1, 0.5]))

        result = panda_osc.compute_control(target)

        # Configuration should have changed (the controller integrates in place).
        assert not np.allclose(result.q, q_before)
        np.testing.assert_allclose(result.q, panda_osc.configuration.q)

    def test_mobile_base_velocity_respects_configured_limits(self, mobile_robot):
        osc = OperationalSpaceController(robot=mobile_robot)
        q = mobile_robot.config.homing_presets[HomingPreset.HOME]
        osc.update_state(q)
        tool_target = mobile_robot.get_tool_command_pose(q)
        base_target = mobile_robot.get_base_pose(q)
        assert base_target is not None
        base_target.translation[0] += 1.0

        result = osc.compute_control(tool_target, base_target_pose=base_target)

        root_v_slice = mobile_robot.get_root_v_slice()
        base_config = mobile_robot.config.base
        assert root_v_slice is not None
        assert base_config is not None
        limits = base_config.velocity_limits
        root_velocity = result.v[root_v_slice]
        assert abs(root_velocity[0]) <= limits.linear + 1e-9
        assert abs(root_velocity[1]) <= limits.linear + 1e-9
        assert abs(root_velocity[2]) <= limits.angular + 1e-9

    def test_tool_task_target_is_updated(self, panda_osc, panda_robot):
        panda_osc.update_state(panda_robot.config.homing_presets[HomingPreset.HOME])
        target = pin.SE3(np.eye(3), np.array([0.4, 0.2, 0.5]))

        panda_osc.compute_control(target)

        actual_target = panda_osc.tasks[TaskName.TOOL].transform_target_to_world
        np.testing.assert_allclose(actual_target.translation, target.translation, atol=1e-9)

    def test_mobile_tool_target_is_relative_to_base(self, mobile_osc, mobile_robot):
        mobile_osc.update_state(mobile_robot.config.homing_presets[HomingPreset.HOME])
        target = mobile_robot.get_tool_command_pose(
            mobile_robot.config.homing_presets[HomingPreset.HOME]
        )

        mobile_osc.compute_control(target)

        task = mobile_osc.tasks[TaskName.TOOL]
        assert isinstance(task, RelativeFrameTask)
        np.testing.assert_allclose(
            task.transform_target_to_root.translation,
            target.translation,
            atol=1e-9,
        )

    def test_mobile_tool_task_does_not_use_floating_base_velocity(self, mobile_osc, mobile_robot):
        mobile_osc.update_state(mobile_robot.config.homing_presets[HomingPreset.HOME])
        task = mobile_osc.tasks[TaskName.TOOL]
        root_v_slice = mobile_robot.get_root_v_slice()
        assert isinstance(task, RelativeFrameTask)
        assert mobile_osc.configuration is not None
        assert root_v_slice is not None

        task.set_target(
            mobile_robot.get_tool_command_pose(
                mobile_robot.config.homing_presets[HomingPreset.HOME]
            )
        )
        jacobian = task.compute_jacobian(mobile_osc.configuration)

        np.testing.assert_allclose(jacobian[:, root_v_slice], 0.0, atol=1e-9)

    def test_gripper_positions_override_q(self, panda_osc, panda_robot):
        """When gripper_positions are provided, they replace those indices in q."""
        panda_osc.update_state(panda_robot.config.homing_presets[HomingPreset.HOME])
        target = pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.4]))
        gripper = np.array([0.0123])  # panda has gripper_joint_indices=[7]

        result = panda_osc.compute_control(target, gripper_positions=gripper)

        gripper_idx = panda_robot.config.gripper_joint_indices[0]
        assert result.q[gripper_idx] == pytest.approx(gripper[0])

    def test_gripper_positions_ignored_when_no_indices(self, panda_robot):
        """When gripper_joint_indices is None, gripper_positions are silently ignored."""
        config = replace(panda_robot.config, gripper_joint_indices=None)
        robot = Robot.__new__(Robot)
        robot.__dict__.update(panda_robot.__dict__)
        robot._config = config

        osc = OperationalSpaceController(robot=robot)
        osc.update_state(config.homing_presets[HomingPreset.HOME])
        target = pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.4]))

        gripper_pos = 99.0

        # Should not raise and should not modify any joints from gripper data.
        result = osc.compute_control(target, gripper_positions=np.array([gripper_pos]))
        assert not np.any(result.q == gripper_pos)

    def test_gripper_override_uses_position_index_not_joint_index(self, mobile_osc, mobile_robot):
        """Regression: gripper override must write via position index, not joint index.

        On the mobile robot the planar base joint shifts the q layout, so the
        gripper's joint index (11) differs from its position index in q (17).
        Indexing q with the raw joint index writes to the wrong slot — this
        test fails if that regresses.
        """
        joint_idx = mobile_robot.config.gripper_joint_indices[0]
        position_idx = mobile_robot.joint_idx_to_position_idx(joint_idx)
        assert joint_idx != position_idx, (
            "Test setup expects joint_idx != position_idx for the mobile robot's "
            "gripper joint; otherwise this test cannot detect the bug."
        )

        mobile_osc.update_state(mobile_robot.config.homing_presets[HomingPreset.HOME])
        target = pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.4]))
        gripper_value = mobile_robot.config.homing_presets[HomingPreset.HOME][position_idx] + 0.5

        result = mobile_osc.compute_control(target, gripper_positions=np.array([gripper_value]))

        assert result.q[position_idx] == pytest.approx(gripper_value)
        assert result.q[joint_idx] != pytest.approx(gripper_value)

    def test_base_target_pose_sets_base_task(self, mobile_osc, mobile_robot):
        """For mobile robots, base_target_pose updates the BASE task target."""
        mobile_osc.update_state(mobile_robot.config.homing_presets[HomingPreset.HOME])
        base_target = pin.SE3(np.eye(3), np.array([1.0, 0.5, 0.0]))
        tool_target = pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.4]))

        mobile_osc.compute_control(tool_target, base_target_pose=base_target)

        actual_base_target = mobile_osc.tasks[TaskName.BASE].transform_target_to_world
        np.testing.assert_allclose(
            actual_base_target.translation, base_target.translation, atol=1e-9
        )
