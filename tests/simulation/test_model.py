"""Tests for reusable MuJoCo model composition and explicit bindings."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.config.robot.triskel import (
    PROVISIONAL_TRISKEL_GRIPPER_CONFIG,
    TRISKEL_CONFIG,
)
from humanoid.robots.base import Robot
from humanoid.simulation.binding import resolve_mujoco_robot_binding
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.model import ROOT_JOINT_NAMES, build_mujoco_spec
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import RobotConfig, RobotJointCommand

MIMIC_SOURCE_TARGET = 1.0
MIMIC_CONSTRAINT_TOLERANCE = 3e-5


@pytest.fixture(scope="module")
def triskel_model():
    robot = Robot(TRISKEL_CONFIG)
    return robot, build_mujoco_spec(robot).compile()


@pytest.mark.parametrize(
    "mesh_name", ["lower_plate", "upper_plate", "wheel", "wheel_mount", "servo"]
)
def test_complete_obj_surfaces_survive_mujoco_import(triskel_model, mesh_name):
    robot, model = triskel_model
    source = robot.urdf_path.parent.parent / "meshes" / f"{mesh_name}.obj"
    records = [line.split() for line in source.read_text().splitlines()]
    # These CAD exports contain many separate objects, with triangulated faces.
    faces = [record for record in records if record and record[0] == "f"]
    assert all(len(face) == 4 for face in faces)  # noqa: PLR2004 - f plus three vertices
    assert int(model.mesh(mesh_name).facenum[0]) == len(faces)


@pytest.mark.parametrize("link", ["finger_1_link", "finger_2_link"])
def test_finger_meshes_remain_visible_without_replacing_contact_pads(triskel_model, link):
    _, model = triskel_model
    geoms = np.flatnonzero(model.geom_bodyid == model.body(link).id)
    visible = [index for index in geoms if model.geom_rgba[index, 3] > 0]
    assert len(visible) == 1
    mesh = visible[0]
    assert model.geom_type[mesh] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_contype[mesh] == model.geom_conaffinity[mesh] == 0
    contacts = [index for index in geoms if model.geom_contype[index]]
    assert len(contacts) == 3  # noqa: PLR2004 - carriage, stem, contact pad
    assert all(model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX for index in contacts)
    assert all(model.geom_rgba[index, 3] == 0 for index in contacts)


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


def test_triskel_gripper_profile_controls_motor_force_contacts_and_constraint_solver():
    profile = replace(
        PROVISIONAL_TRISKEL_GRIPPER_CONFIG,
        motor_torque_limit=0.6,
        friction=(0.6, 0.006, 0.0002),
        contact_dimension=3,
        contact_solref=(0.012, 1.2),
        contact_solimp=(0.9, 0.98, 0.002, 0.4, 2.0),
        coupling_solref=(0.006, 1.1),
        coupling_solimp=(0.99, 0.999, 0.002, 0.4, 2.0),
    )
    assert TRISKEL_CONFIG.gripper is not None
    engine = NativeMujocoEngine(
        replace(TRISKEL_CONFIG, gripper=replace(TRISKEL_CONFIG.gripper, simulation=profile))
    )
    target = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy()
    target[-1] = MIMIC_SOURCE_TARGET
    engine.apply_joint_command(RobotJointCommand(timestamp=0.0, joint_positions=target))
    engine.step(1)
    motor = engine.model.actuator("gripper_1")
    assert engine.data.actuator_force[motor.id] == pytest.approx(profile.motor_torque_limit)
    for link in ("finger_1_link", "finger_2_link"):
        for index in range(3):
            contact = engine.model.geom(f"{link}_contact_{index}")
            assert contact.friction == pytest.approx(profile.friction)
            assert contact.condim == profile.contact_dimension
            assert contact.solref == pytest.approx(profile.contact_solref)
            assert contact.solimp == pytest.approx(profile.contact_solimp)
    for joint in ("gripper_2", "gripper_3"):
        coupling = engine.model.equality(f"{joint}_mimics_gripper_1")
        assert coupling.solref == pytest.approx(profile.coupling_solref)
        assert coupling.solimp == pytest.approx(profile.coupling_solimp)


@pytest.mark.parametrize(
    "profile",
    [None, replace(PROVISIONAL_TRISKEL_GRIPPER_CONFIG, motor_torque_limit=None)],
    ids=["native-contact-defaults", "provisional-contacts"],
)
def test_urdf_motor_limit_can_be_used_without_removing_collision_shapes_or_coupling(profile):
    assert TRISKEL_CONFIG.gripper is not None
    engine = NativeMujocoEngine(
        replace(TRISKEL_CONFIG, gripper=replace(TRISKEL_CONFIG.gripper, simulation=profile))
    )
    robot = engine.robot
    motor_index = robot.joint_idx_to_velocity_idx(robot.joint_name_to_idx("gripper_1"))
    limit = robot.model.effortLimit[motor_index]
    assert engine.model.actuator("gripper_1").forcerange == pytest.approx([-limit, limit])
    for link in ("finger_1_link", "finger_2_link"):
        for index in range(3):
            contact = engine.model.geom(f"{link}_contact_{index}")
            assert contact.type == mujoco.mjtGeom.mjGEOM_BOX
            assert contact.contype != 0
    for joint, multiplier in (("gripper_2", -0.0115), ("gripper_3", 0.0115)):
        coupling = engine.model.equality(f"{joint}_mimics_gripper_1")
        assert coupling.active0
        assert coupling.data[:5] == pytest.approx([0, multiplier, 0, 0, 0])


def _joint_position(engine: NativeMujocoEngine, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return float(engine.data.qpos[engine.model.jnt_qposadr[joint_id]])
