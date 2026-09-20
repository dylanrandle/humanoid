from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest
from pink.tasks import FrameTask, LowAccelerationTask, RelativeFrameTask

from humanoid.config import ROBOT_CONFIGS
from humanoid.controllers.manipulability import ManipulabilityTask
from humanoid.controllers.operational_space import (
    OperationalSpaceController,
    clamp_cartesian_velocity,
)
from humanoid.robots.base import Robot
from humanoid.types.controllers import ControlResult, OperationalSpaceConfig, TaskName
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import CartesianVelocity, CartesianVelocityLimits


@pytest.fixture(scope="module")
def panda_robot() -> Robot:
    """Fixed-base 7-DOF Panda arm — no base frame configured."""
    return Robot(ROBOT_CONFIGS["panda"])


@pytest.fixture(scope="module")
def mobile_robot() -> Robot:
    """Mobile manipulator with a base frame configured."""
    return Robot(ROBOT_CONFIGS["triskel"])


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
        assert TaskName.MANIPULABILITY in panda_osc.tasks
        assert TaskName.DAMPING in panda_osc.tasks
        assert set(panda_osc.tasks) == {
            TaskName.TOOL,
            TaskName.MANIPULABILITY,
            TaskName.DAMPING,
        }

    def test_tasks_for_mobile_robot_are_arm_only(self, mobile_osc):
        """A mobile robot tracks its tool relative to, but does not drive, its base."""
        assert isinstance(mobile_osc.tasks[TaskName.TOOL], RelativeFrameTask)
        assert set(mobile_osc.tasks) == {
            TaskName.TOOL,
            TaskName.MANIPULABILITY,
            TaskName.DAMPING,
        }
        controlled_names = [
            mobile_osc.robot.joint_idx_to_name(index)
            for index in mobile_osc.controlled_joint_indices
        ]
        assert controlled_names == [f"arm_{index}" for index in range(1, 8)]

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

    def test_cartesian_feedforward_is_clamped_by_vector_norm(self):
        velocity = CartesianVelocity(
            linear=np.array([3.0, 4.0, 0.0]),
            angular=np.array([0.0, 0.0, -2.0]),
        )

        bounded = clamp_cartesian_velocity(
            velocity,
            CartesianVelocityLimits(linear=1.0, angular=0.5),
        )

        np.testing.assert_allclose(bounded.linear, [0.6, 0.8, 0.0])
        np.testing.assert_allclose(bounded.angular, [0.0, 0.0, -0.5])

    def test_configuration_is_none_initially(self, panda_osc):
        """configuration is deferred until first update_state call."""
        assert panda_osc.configuration is None

    def test_manipulability_configuration_is_used(self, panda_robot):
        config = OperationalSpaceConfig(
            manipulability_cost=0.02,
            manipulability_regularization=1e-4,
        )

        osc = OperationalSpaceController(robot=panda_robot, config=config)

        task = osc.tasks[TaskName.MANIPULABILITY]
        assert isinstance(task, ManipulabilityTask)
        assert task.cost == config.manipulability_cost
        assert task.regularization == config.manipulability_regularization

    def test_zero_manipulability_cost_disables_task(self, panda_robot):
        osc = OperationalSpaceController(
            robot=panda_robot, config=OperationalSpaceConfig(manipulability_cost=0.0)
        )

        assert TaskName.MANIPULABILITY not in osc.tasks

    @pytest.mark.parametrize("cost", [-1.0, np.inf, np.nan])
    def test_invalid_manipulability_cost_is_rejected(self, cost):
        with pytest.raises(ValueError, match="Manipulability cost"):
            OperationalSpaceConfig(manipulability_cost=cost)

    @pytest.mark.parametrize("regularization", [0.0, -1.0, np.inf, np.nan])
    def test_invalid_manipulability_regularization_is_rejected(self, regularization):
        with pytest.raises(ValueError, match="Manipulability regularization"):
            OperationalSpaceConfig(manipulability_regularization=regularization)

    def test_optional_low_acceleration_task_is_enabled(self, panda_robot):
        config = OperationalSpaceConfig(low_acceleration_cost=0.1)

        osc = OperationalSpaceController(robot=panda_robot, config=config)

        assert isinstance(osc.tasks[TaskName.LOW_ACCELERATION], LowAccelerationTask)

    def test_low_acceleration_mask_is_independent_from_damping_mask(self, panda_robot):
        low_acceleration_mask = np.array([1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0])
        config = OperationalSpaceConfig(
            damping_cost=0.2,
            damping_mask=0.0,
            low_acceleration_cost=0.3,
            low_acceleration_mask=low_acceleration_mask,
        )

        osc = OperationalSpaceController(robot=panda_robot, config=config)

        damping_cost = osc.tasks[TaskName.DAMPING].cost
        np.testing.assert_allclose(damping_cost, np.zeros_like(damping_cost))
        expected_low_acceleration_cost = np.zeros_like(osc.tasks[TaskName.LOW_ACCELERATION].cost)
        expected_low_acceleration_cost[osc._arm_task_indices] = (
            config.low_acceleration_cost * low_acceleration_mask
        )
        np.testing.assert_allclose(
            osc.tasks[TaskName.LOW_ACCELERATION].cost,
            expected_low_acceleration_cost,
        )

    def test_triskel_motion_limits_construct_for_mobile_model(self, mobile_robot):
        config = mobile_robot.config.operational_space_config
        assert config is not None

        osc = OperationalSpaceController(robot=mobile_robot, config=config)

        assert osc._acceleration_limit is not None

        q = mobile_robot.config.homing_presets[HomingPreset.HOME].copy()
        osc.update_state(q)
        target = mobile_robot.get_tool_command_pose(q)
        target.translation[0] += 0.01

        result = osc.compute_control(target, dt=config.dt)

        assert np.any(np.abs(result.v[osc.controlled_v_indices]) > 0.0)

    def test_collision_barrier_uses_configured_safe_displacement_gain(self, mobile_robot):
        configured_gain = 0.025
        config = OperationalSpaceConfig(
            avoid_collisions=True,
            collision_safe_displacement_gain=configured_gain,
        )

        osc = OperationalSpaceController(robot=mobile_robot, config=config)

        assert len(osc.barriers) == 1
        assert osc.barriers[0].safe_displacement_gain == pytest.approx(configured_gain)

    @pytest.mark.parametrize("gain", [-1.0, np.inf, np.nan])
    def test_invalid_collision_safe_displacement_gain_is_rejected(self, gain):
        with pytest.raises(ValueError, match="safe-displacement"):
            OperationalSpaceConfig(collision_safe_displacement_gain=gain)

    @pytest.mark.parametrize("field", ["joint_velocity_limit", "joint_acceleration_limit"])
    @pytest.mark.parametrize("limit", [0.0, -1.0, np.inf, np.nan])
    def test_invalid_joint_motion_limit_is_rejected(self, field, limit):
        with pytest.raises(ValueError, match="limits"):
            OperationalSpaceConfig(**{field: limit})


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
    @pytest.mark.parametrize("robot_fixture", ["panda_robot", "mobile_robot"])
    def test_holding_tool_pose_increases_manipulability(self, request, robot_fixture):
        minimum_arm_speed = 1e-6
        tool_error_tolerance = 1e-4
        robot = request.getfixturevalue(robot_fixture)
        osc = OperationalSpaceController(robot=robot)
        q = robot.config.homing_presets[HomingPreset.HOME].copy()
        # Break symmetric home postures so the redundant arm motion has a
        # nonzero manipulability gradient while the tool holds its pose.
        q[osc.controlled_q_indices[2]] += 0.2
        osc.update_state(q)
        target = robot.get_tool_command_pose(q)
        task = osc.tasks[TaskName.MANIPULABILITY]
        before = task.compute_log_manipulability(osc.configuration)

        for _ in range(20):
            result = osc.compute_control(target)

        assert np.linalg.norm(result.v[osc.controlled_v_indices]) > minimum_arm_speed
        assert task.compute_log_manipulability(osc.configuration) > before
        tool_error = osc.tasks[TaskName.TOOL].compute_error(osc.configuration)
        assert np.linalg.norm(tool_error) < tool_error_tolerance

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

    def test_mobile_base_wheels_and_gripper_do_not_move(self, mobile_robot):
        osc = OperationalSpaceController(robot=mobile_robot)
        q = mobile_robot.config.homing_presets[HomingPreset.HOME].copy()
        osc.update_state(q)
        tool_target = mobile_robot.get_tool_command_pose(q)
        tool_target.translation[0] += 0.05

        result = osc.compute_control(tool_target)

        root_q_slice = mobile_robot.get_root_q_slice()
        root_v_slice = mobile_robot.get_root_v_slice()
        wheel_joint_indices = mobile_robot.get_wheel_joint_indices()
        wheel_q_indices = mobile_robot.get_joint_position_indices(wheel_joint_indices)
        wheel_v_indices = mobile_robot.get_joint_velocity_indices(wheel_joint_indices)
        gripper_q_indices = mobile_robot.get_gripper_position_indices()
        gripper_v_indices = mobile_robot.get_joint_velocity_indices(
            mobile_robot.get_gripper_joint_indices()
        )
        assert root_q_slice is not None
        assert root_v_slice is not None
        np.testing.assert_allclose(result.q[root_q_slice], q[root_q_slice])
        np.testing.assert_allclose(result.q[wheel_q_indices], q[wheel_q_indices])
        np.testing.assert_allclose(result.q[gripper_q_indices], q[gripper_q_indices])
        np.testing.assert_array_equal(result.v[root_v_slice], 0.0)
        np.testing.assert_array_equal(result.v[wheel_v_indices], 0.0)
        np.testing.assert_array_equal(result.v[gripper_v_indices], 0.0)

    def test_tool_task_target_is_updated(self, panda_osc, panda_robot):
        panda_osc.update_state(panda_robot.config.homing_presets[HomingPreset.HOME])
        target = pin.SE3(np.eye(3), np.array([0.4, 0.2, 0.5]))

        panda_osc.compute_control(target)

        actual_target = panda_osc.tasks[TaskName.TOOL].transform_target_to_world
        np.testing.assert_allclose(actual_target.translation, target.translation, atol=1e-9)

    def test_velocity_feedforward_previews_the_tool_target(self, panda_osc, panda_robot):
        panda_osc.update_state(panda_robot.config.homing_presets[HomingPreset.HOME])
        target = panda_robot.get_tool_command_pose(
            panda_robot.config.homing_presets[HomingPreset.HOME]
        )
        velocity = CartesianVelocity(
            linear=np.array([0.1, -0.2, 0.3]),
            angular=np.array([0.0, 0.0, 0.4]),
        )
        dt = 0.05

        panda_osc.compute_control(target, dt=dt, target_velocity=velocity)

        actual_target = panda_osc.tasks[TaskName.TOOL].transform_target_to_world
        np.testing.assert_allclose(
            actual_target.translation,
            target.translation + velocity.linear * dt,
        )
        np.testing.assert_allclose(
            actual_target.rotation,
            pin.exp3(velocity.angular * dt) @ target.rotation,
        )

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

    def test_fixed_robot_gripper_does_not_move(self, panda_osc, panda_robot):
        panda_osc.update_state(panda_robot.config.homing_presets[HomingPreset.HOME])
        q_before = panda_osc.configuration.q.copy()
        target = pin.SE3(np.eye(3), np.array([0.4, 0.0, 0.4]))
        result = panda_osc.compute_control(target)

        gripper_indices = panda_robot.get_gripper_position_indices()
        np.testing.assert_allclose(result.q[gripper_indices], q_before[gripper_indices])
        np.testing.assert_array_equal(result.v[gripper_indices], 0.0)

    def test_hard_acceleration_limit_bounds_joint_velocity_change(self, panda_robot):
        acceleration_limit = 0.5
        dt = 0.1
        osc = OperationalSpaceController(
            robot=panda_robot,
            config=OperationalSpaceConfig(
                dt=dt,
                joint_acceleration_limit=acceleration_limit,
            ),
        )
        q = panda_robot.config.homing_presets[HomingPreset.HOME].copy()
        osc.update_state(q)
        target = panda_robot.get_tool_command_pose(q)
        target.translation[0] += 0.1

        first = osc.compute_control(target, dt=dt)
        second = osc.compute_control(target, dt=dt)
        arm_v_indices = panda_robot.get_joint_velocity_indices(panda_robot.get_arm_joint_indices())

        max_velocity_change = acceleration_limit * dt + 1e-6
        assert np.max(np.abs(first.v[arm_v_indices])) <= max_velocity_change
        assert np.max(np.abs(second.v[arm_v_indices] - first.v[arm_v_indices])) <= (
            max_velocity_change
        )

    def test_acceleration_limit_with_changing_control_period(self, mobile_robot):
        config = replace(mobile_robot.config.operational_space_config, manipulability_cost=0.0)
        osc = OperationalSpaceController(mobile_robot, config)
        q = mobile_robot.config.homing_presets[HomingPreset.HOME].copy()
        osc.update_state(q)
        target = mobile_robot.get_tool_command_pose(q).copy()
        target.translation[0] += 0.05
        previous = np.zeros(mobile_robot.model.nv)
        for dt in [0.035, 0.035, 0.03] * 10:
            result = osc.compute_control(target, dt=dt)
            assert result.diagnostics is not None and result.diagnostics.succeeded
            acceleration = (result.v - previous)[osc.controlled_v_indices] / dt
            assert np.abs(acceleration).max() <= config.joint_acceleration_limit + 1e-6
            previous = result.v

    def test_solve_failure_is_explicit_in_result(self, panda_osc, panda_robot, monkeypatch):
        q = panda_robot.config.homing_presets[HomingPreset.HOME].copy()
        panda_osc.update_state(q)

        def fail(*args, **kwargs):
            raise RuntimeError("injected infeasible QP")

        monkeypatch.setattr("pink.solve_ik", fail)
        result = panda_osc.compute_control(panda_robot.get_tool_command_pose(q))
        assert result.diagnostics is not None
        assert not result.diagnostics.succeeded
        assert result.diagnostics.error == "injected infeasible QP"
        assert result.diagnostics.duration_s >= 0.0
        np.testing.assert_array_equal(result.q, q)
        np.testing.assert_array_equal(result.v, 0.0)

    def test_configured_joint_velocity_limit_is_enforced(self, panda_robot):
        velocity_limit = 0.1
        osc = OperationalSpaceController(
            robot=panda_robot,
            config=OperationalSpaceConfig(joint_velocity_limit=velocity_limit),
        )
        q = panda_robot.config.homing_presets[HomingPreset.HOME].copy()
        osc.update_state(q)
        target = panda_robot.get_tool_command_pose(q)
        target.translation[0] += 0.5

        result = osc.compute_control(target)

        arm_v_indices = panda_robot.get_joint_velocity_indices(panda_robot.get_arm_joint_indices())
        assert np.max(np.abs(result.v[arm_v_indices])) <= velocity_limit + 1e-6
