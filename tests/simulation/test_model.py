"""Tests for reusable MuJoCo model composition and explicit bindings."""

import mujoco
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.config.robot.elrobot import ELROBOT_CONFIG
from humanoid.robots.base import Robot
from humanoid.simulation.binding import resolve_mujoco_robot_binding
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.model import ROOT_JOINT_NAMES, build_mujoco_spec
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import RobotConfig, RobotJointCommand

MIMIC_SOURCE_TARGET = 1.0
MIMIC_CONSTRAINT_TOLERANCE = 3e-5


@pytest.mark.parametrize("robot_config", ROBOT_CONFIGS.values(), ids=lambda config: config.name)
def test_every_configured_robot_builds_a_named_mujoco_model(robot_config: RobotConfig):
    robot = Robot(robot_config)

    model = build_mujoco_spec(robot).compile()
    binding = resolve_mujoco_robot_binding(model, robot)

    assert tuple(joint.name for joint in binding.joints) == tuple(robot.actuator_joint_names)
    assert all(joint.actuator_id >= 0 for joint in binding.joints)
    if robot_config.base is None:
        assert binding.root is None
    else:
        assert binding.root is not None
        assert (
            tuple(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                for joint_id in binding.root.joint_ids
            )
            == ROOT_JOINT_NAMES
        )


@pytest.mark.parametrize("robot_config", ROBOT_CONFIGS.values(), ids=lambda config: config.name)
def test_mujoco_controlled_joint_limits_match_pinocchio(robot_config: RobotConfig):
    robot = Robot(robot_config)
    model = build_mujoco_spec(robot).compile()
    binding = resolve_mujoco_robot_binding(model, robot)

    for joint in binding.joints:
        if not model.jnt_limited[joint.joint_id]:
            # Continuous joints have one scalar MuJoCo angle but two Pinocchio
            # configuration coordinates [cos(theta), sin(theta)].
            continue
        robot_joint_idx = robot.joint_name_to_idx(joint.name)
        position_idx = robot.joint_idx_to_position_idx(robot_joint_idx)
        assert model.jnt_range[joint.joint_id] == pytest.approx(
            [
                robot.model.lowerPositionLimit[position_idx],
                robot.model.upperPositionLimit[position_idx],
            ]
        )


def test_elrobot_gripper_actuation_enforces_mimic_multipliers_and_offsets():
    engine = NativeMujocoEngine(ELROBOT_CONFIG)
    target = ELROBOT_CONFIG.homing_presets[HomingPreset.HOME].copy()
    target[-1] = MIMIC_SOURCE_TARGET

    engine.apply_joint_command(RobotJointCommand(timestamp=0.0, joint_positions=target))
    engine.step(1000)

    source_position = _joint_position(engine, "gripper_1")
    assert source_position == pytest.approx(MIMIC_SOURCE_TARGET, abs=0.01)
    for joint_name, multiplier, offset in (
        ("gripper_2", -0.0115, 0.0),
        ("gripper_3", 0.0115, 0.0),
    ):
        assert _joint_position(engine, joint_name) == pytest.approx(
            offset + multiplier * source_position,
            abs=MIMIC_CONSTRAINT_TOLERANCE,
        )


def _joint_position(engine: NativeMujocoEngine, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return float(engine.data.qpos[engine.model.jnt_qposadr[joint_id]])
