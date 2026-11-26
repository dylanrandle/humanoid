"""Tests for reusable MuJoCo model composition and explicit bindings."""

from dataclasses import replace
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.config.actuator import STS3215_345_OUTPUT_ARMATURE
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.robots.base import Robot
from humanoid.simulation.binding import resolve_mujoco_robot_binding
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.model import ROOT_JOINT_NAMES, build_mujoco_spec
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import RobotConfig, RobotJointCommand

SOFT_FINGER_CONTACT_DIM = 4
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


def test_triskel_gripper_actuation_enforces_mimic_multipliers_and_offsets():
    engine = NativeMujocoEngine(TRISKEL_CONFIG)
    target = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy()
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


def test_triskel_applies_reflected_inertia_once_at_each_sts3215_output():
    robot = Robot(TRISKEL_CONFIG)
    model = build_mujoco_spec(robot).compile()
    for name in robot.actuator_joint_names:
        shaft = robot.model.joints[robot.model.getJointId(name)]
        assert robot.model.armature[shaft.idx_v] == pytest.approx(STS3215_345_OUTPUT_ARMATURE)
        assert robot.model.velocityLimit[shaft.idx_v] == pytest.approx(52 * 2 * np.pi / 60)
        dof = model.joint(name).dofadr[0]
        assert model.dof_armature[dof] == pytest.approx(STS3215_345_OUTPUT_ARMATURE)
        assert model.dof_damping[dof] == pytest.approx(0.05989)
        assert model.dof_frictionloss[dof] == pytest.approx(0.05152)
        assert model.actuator(name).forcerange == pytest.approx([-1.91229675, 1.91229675])
        assert model.actuator(name).gear[0] == 1.0
    for rack in ("gripper_2", "gripper_3"):
        assert model.dof_armature[model.joint(rack).dofadr[0]] == 0.0


def test_position_and_velocity_drives_use_simulation_gains():
    robot = Robot(TRISKEL_CONFIG)
    config = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, position_kp=84.0, velocity_kv=3.5)
    model = build_mujoco_spec(robot, config).compile()
    half_damping = replace(config, position_damping_ratio=config.position_damping_ratio / 2)
    less_damped = build_mujoco_spec(robot, half_damping).compile()

    for name in ("arm_2", "gripper_1"):
        actuator = model.actuator(name)
        assert actuator.gainprm[0] == pytest.approx(config.position_kp)
        assert actuator.biasprm[2] < 0
        assert less_damped.actuator(name).biasprm[2] == pytest.approx(actuator.biasprm[2] / 2)
    assert model.actuator("wheel_1").gainprm[0] == pytest.approx(config.velocity_kv)


def test_triskel_compiles_urdf_masses_and_finger_inertia():
    robot = Robot(TRISKEL_CONFIG)
    model = build_mujoco_spec(robot).compile()
    for link in ET.parse(robot.urdf_path).getroot().findall("link"):
        body = model.body(link.attrib["name"])
        mass = link.find("inertial/mass")
        expected = float(mass.attrib["value"]) if mass is not None else 0.0
        assert body.mass.item() == pytest.approx(expected)
        if link.attrib["name"] in ("finger_1_link", "finger_2_link"):
            inertia = link.find("inertial/inertia")
            assert inertia is not None
            principal = [float(inertia.attrib[axis]) for axis in ("ixx", "iyy", "izz")]
            np.testing.assert_allclose(sorted(body.inertia), sorted(principal))


def test_triskel_fin_ray_contact_moves_with_the_intact_finger():
    model = build_mujoco_spec(Robot(TRISKEL_CONFIG)).compile()
    for n in (1, 2):
        finger = model.body(f"finger_{n}_link")
        contact = model.geom(f"finger_{n}_fin_contact")
        assert model.geom_bodyid[contact.id] == finger.id
        assert model.geom_condim[contact.id] == SOFT_FINGER_CONTACT_DIM
        assert model.jnt_type[finger.jntadr[0]] == mujoco.mjtJoint.mjJNT_SLIDE
        visual_mesh = model.mesh(f"finger_{n}_fin").id
        visuals = (model.geom_dataid == visual_mesh) & (model.geom_contype == 0)
        assert model.geom_bodyid[visuals].tolist() == [finger.id]


def test_triskel_solver_options_apply_without_pad_overrides():
    triskel = replace(TRISKEL_CONFIG, geom_contacts={})
    model = build_mujoco_spec(Robot(triskel)).compile()
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_DIAGEXACT


def test_contact_settings_require_a_matching_collision_geom():
    pad = TRISKEL_CONFIG.geom_contacts["finger_1_fin_contact"]
    robot = Robot(replace(TRISKEL_CONFIG, geom_contacts={"missing_pad": pad}))
    with pytest.raises(ValueError, match="No robot collision geometry named 'missing_pad'"):
        build_mujoco_spec(robot)
