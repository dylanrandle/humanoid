"""Physics and state-mapping tests for the native MuJoCo engine."""

import math

import mujoco
import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.config.robot.so101 import SO101_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.actuator import ActuatorControlMode, ActuatorEffortSource
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import RobotConfig, RobotJointCommand

TEST_TIMESTAMP = 12.0
MAXIMUM_HOME_ERROR = 0.03
MINIMUM_ROOT_RESPONSE = 0.05
MAXIMUM_STOPPED_ROOT_VELOCITY = 0.01
MAXIMUM_ORTHOGONAL_ROOT_DRIFT = 0.02


@pytest.mark.parametrize("robot_config", ROBOT_CONFIGS.values(), ids=lambda config: config.name)
def test_reset_publishes_the_configured_home_state(robot_config: RobotConfig):
    engine = NativeMujocoEngine(robot_config)

    state = engine.read_robot_state(timestamp=TEST_TIMESTAMP)

    assert state.timestamp == TEST_TIMESTAMP
    assert state.joint_positions == pytest.approx(robot_config.homing_presets[HomingPreset.HOME])
    assert state.joint_velocities == pytest.approx(np.zeros(engine.robot.model.nv))
    assert state.actuator_temperatures.shape == (len(engine.binding.joints),)
    assert state.effort_source is ActuatorEffortSource.SIMULATION
    assert state.joint_efforts is not None
    assert state.joint_efforts.shape == (engine.robot.model.nv,)
    for joint in engine.binding.joints:
        joint_index = engine.robot.joint_name_to_idx(joint.name)
        effort_index = engine.robot.joint_idx_to_velocity_idx(joint_index)
        assert state.joint_efforts[effort_index] == pytest.approx(
            engine.data.qfrc_actuator[joint.qvel_address]
        )


@pytest.mark.parametrize("offset", [-0.1, 0.1])
def test_position_command_effort_preserves_direction(offset):
    engine = NativeMujocoEngine(SO101_CONFIG)
    joint = engine.binding.joints[0]
    joint_index = engine.robot.joint_name_to_idx(joint.name)
    position_index = engine.robot.joint_idx_to_position_idx(joint_index)
    effort_index = engine.robot.joint_idx_to_velocity_idx(joint_index)
    target = SO101_CONFIG.homing_presets[HomingPreset.HOME].copy()
    target[position_index] += offset

    engine.apply_joint_command(RobotJointCommand(timestamp=0.0, joint_positions=target))
    mujoco.mj_forward(engine.model, engine.data)
    state = engine.read_robot_state(timestamp=TEST_TIMESTAMP)

    assert state.joint_efforts is not None
    assert state.joint_efforts[effort_index] * offset > 0.0
    assert state.joint_efforts[effort_index] == pytest.approx(
        engine.data.qfrc_actuator[joint.qvel_address]
    )


@pytest.mark.parametrize("robot_config", ROBOT_CONFIGS.values(), ids=lambda config: config.name)
def test_home_state_remains_stable_under_physics(robot_config: RobotConfig):
    engine = NativeMujocoEngine(robot_config)

    engine.step(2000)
    state = engine.read_robot_state(timestamp=0.0)

    assert np.isfinite(state.joint_positions).all()
    assert np.isfinite(state.joint_velocities).all()
    assert (
        np.max(np.abs(state.joint_positions - robot_config.homing_presets[HomingPreset.HOME]))
        < MAXIMUM_HOME_ERROR
    )
    assert not any(warning.number for warning in engine.data.warning)


def test_position_command_moves_a_fixed_base_robot():
    engine = NativeMujocoEngine(SO101_CONFIG)
    target = SO101_CONFIG.homing_presets[HomingPreset.HOME].copy()
    target[0] += 0.2

    engine.apply_joint_command(RobotJointCommand(timestamp=0.0, joint_positions=target))
    engine.step(1000)
    state = engine.read_robot_state(timestamp=1.0)

    assert state.joint_positions[0] == pytest.approx(target[0], abs=0.01)


def test_stop_clears_velocity_targets_but_preserves_position_hold():
    engine = NativeMujocoEngine(TRISKEL_CONFIG)
    target = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy()
    engine.apply_joint_command(RobotJointCommand(0.0, target, np.full(engine.robot.model.nv, 0.1)))
    engine.step(20)
    control = engine.data.ctrl.copy()
    engine.stop_velocity_actuators()
    for joint in engine.binding.joints:
        if joint.control_mode is ActuatorControlMode.VELOCITY:
            assert control[joint.actuator_id] != 0
            assert engine.data.ctrl[joint.actuator_id] == 0
        else:
            assert engine.data.ctrl[joint.actuator_id] == control[joint.actuator_id]
    assert engine.binding.root is not None
    ids = list(engine.binding.root.actuator_ids)
    assert np.all(control[ids] != 0)
    assert engine.data.ctrl[ids] == pytest.approx(0)


def test_mobile_root_follows_the_commanded_generalized_velocity():
    engine = NativeMujocoEngine(TRISKEL_CONFIG)
    velocities = np.zeros(engine.robot.model.nv)
    root_v_slice = engine.robot.get_root_v_slice()
    assert root_v_slice is not None
    velocities[root_v_slice] = [0.1, 0.0, 0.0]

    engine.apply_joint_command(
        RobotJointCommand(
            timestamp=0.0,
            joint_positions=TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy(),
            joint_velocities=velocities,
        )
    )
    engine.step(1000)
    state = engine.read_robot_state(timestamp=1.0)
    root_q_slice = engine.robot.get_root_q_slice()
    assert root_q_slice is not None

    assert state.joint_positions[root_q_slice][0] > MINIMUM_ROOT_RESPONSE
    assert state.joint_velocities[root_v_slice][0] > MINIMUM_ROOT_RESPONSE

    engine.stop_velocity_actuators()
    engine.step(1000)
    stopped = engine.read_robot_state(timestamp=2.0)
    assert abs(stopped.joint_velocities[root_v_slice][0]) < MAXIMUM_STOPPED_ROOT_VELOCITY


def test_mobile_body_velocity_rotates_into_the_mujoco_world_frame():
    engine = NativeMujocoEngine(TRISKEL_CONFIG)
    root = engine.binding.root
    root_q_slice = engine.robot.get_root_q_slice()
    root_v_slice = engine.robot.get_root_v_slice()
    assert root is not None
    assert root_q_slice is not None
    assert root_v_slice is not None
    initial_x, initial_y = engine.data.qpos[list(root.qpos_addresses[:2])]
    engine.data.qpos[root.qpos_addresses[2]] = math.pi / 2.0
    mujoco.mj_forward(engine.model, engine.data)
    velocities = np.zeros(engine.robot.model.nv)
    velocities[root_v_slice] = [0.1, 0.0, 0.0]

    engine.apply_joint_command(
        RobotJointCommand(
            timestamp=0.0,
            joint_positions=TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy(),
            joint_velocities=velocities,
        )
    )
    engine.step(1000)
    state = engine.read_robot_state(timestamp=1.0)

    assert state.joint_positions[root_q_slice][1] - initial_y > MINIMUM_ROOT_RESPONSE
    assert abs(state.joint_positions[root_q_slice][0] - initial_x) < MAXIMUM_ORTHOGONAL_ROOT_DRIFT
    assert state.joint_velocities[root_v_slice][0] > MINIMUM_ROOT_RESPONSE
    assert abs(state.joint_velocities[root_v_slice][1]) < MAXIMUM_ORTHOGONAL_ROOT_DRIFT


def test_mobile_world_velocity_rotates_back_into_the_pinocchio_body_frame():
    engine = NativeMujocoEngine(TRISKEL_CONFIG)
    root = engine.binding.root
    root_v_slice = engine.robot.get_root_v_slice()
    assert root is not None
    assert root_v_slice is not None
    engine.data.qpos[root.qpos_addresses[2]] = math.pi / 2.0
    engine.data.qvel[list(root.qvel_addresses)] = [0.0, 0.1, 0.0]

    state = engine.read_robot_state(timestamp=1.0)

    assert state.joint_velocities[root_v_slice] == pytest.approx([0.1, 0.0, 0.0])


def test_rejects_wrong_sized_or_non_finite_commands():
    engine = NativeMujocoEngine(SO101_CONFIG)
    home = SO101_CONFIG.homing_presets[HomingPreset.HOME]

    with pytest.raises(ValueError, match=r"5 positions.*requires 6"):
        engine.apply_joint_command(RobotJointCommand(timestamp=0.0, joint_positions=home[:-1]))

    invalid = home.copy()
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="positions must all be finite"):
        engine.apply_joint_command(RobotJointCommand(timestamp=0.0, joint_positions=invalid))


def test_rejects_non_positive_substeps():
    engine = NativeMujocoEngine(SO101_CONFIG)

    with pytest.raises(ValueError, match="substeps must be positive"):
        engine.step(0)


def test_bare_engine_does_not_invent_a_task_scene():
    engine = NativeMujocoEngine(TRISKEL_CONFIG)
    assert engine.model.ncam == 0
    assert mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, "cleanup_bin") == -1
    assert mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1


def test_tasks_can_compose_their_own_scene():
    def add_goal(spec):
        spec.worldbody.add_geom(
            name="custom_goal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1]
        )

    engine = NativeMujocoEngine(TRISKEL_CONFIG, scene_builder=add_goal)
    assert engine.model.geom("custom_goal").id >= 0
    assert mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, "cleanup_bin") == -1
