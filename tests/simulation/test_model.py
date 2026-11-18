"""Tests for reusable MuJoCo model composition and explicit bindings."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.robots.base import Robot
from humanoid.simulation.binding import resolve_mujoco_robot_binding
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.model import ROOT_JOINT_NAMES, build_mujoco_spec
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import RobotConfig, RobotJointCommand

MIMIC_SOURCE_TARGET = 1.0
MIMIC_CONSTRAINT_TOLERANCE = 3e-5
MINIMUM_FINGER_GAP_CLEARANCE = 0.005
MAXIMUM_HOLD_SLIP = 0.01


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


def test_contact_solver_retains_a_static_pinch_and_allows_release():
    """Isolate solver slip from grasp planning with a free block between two jaws."""
    robot = Robot(TRISKEL_CONFIG)
    spec = build_mujoco_spec(robot, replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, camera=None))
    for index, x in enumerate((0.9795, 1.0205)):
        spec.worldbody.add_geom(
            name=f"pinch_jaw_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, 1, 1],
            size=[0.01, 0.04, 0.1],
            friction=[1, 0.005, 0.0001],
            contype=2,
            conaffinity=2,
        )
    body = spec.worldbody.add_body(name="pinched_block", pos=[1, 1, 1])
    body.add_freejoint(name="pinched_block_free")
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0125, 0.025, 0.011],
        mass=0.04,
        friction=[1, 0.005, 0.0001],
        contype=2,
        conaffinity=2,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    address = int(model.joint("pinched_block_free").qposadr[0])
    mujoco.mj_step(model, data, nstep=round(60 / model.opt.timestep))

    # MuJoCo's default pyramidal/ratio=1 contact drifts about 74 mm here.
    assert 1.0 - data.qpos[address + 2] < MAXIMUM_HOLD_SLIP
    held_height = data.qpos[address + 2]
    for index, x in enumerate((0.8, 1.2)):
        model.geom_pos[model.geom(f"pinch_jaw_{index}").id, 0] = x
    mujoco.mj_step(model, data, nstep=round(0.3 / model.opt.timestep))

    assert data.qpos[address + 2] < held_height - 0.1


@pytest.mark.parametrize(
    ("finger", "gap_x", "fin_x"),
    [("finger_1_link", 0.020, 0.030), ("finger_2_link", -0.016, -0.030)],
)
def test_triskel_finger_collision_preserves_the_open_gap(finger, gap_x, fin_x):
    robot = Robot(TRISKEL_CONFIG)
    tool = robot.get_frame_pose(
        "gripper_base_link", TRISKEL_CONFIG.homing_presets[HomingPreset.HOME]
    ).copy()

    def scene(spec):
        body = spec.worldbody.add_body(name="gap_probe_body")
        body.add_freejoint(name="gap_probe_joint")
        body.add_geom(
            name="gap_probe",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.001, 0, 0],
            contype=2,
            conaffinity=1,
        )

    engine = NativeMujocoEngine(
        TRISKEL_CONFIG,
        replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, camera=None),
        configure_scene=scene,
    )
    model, data = engine.model, engine.data
    probe = model.geom("gap_probe").id
    probe_position = model.joint("gap_probe_joint").qposadr[0]
    finger_geoms = [
        index
        for index in range(model.ngeom)
        if model.body(model.geom_bodyid[index]).name == finger and model.geom_contype[index]
    ]
    assert finger_geoms
    for x, in_gap in ((gap_x, True), (fin_x, False)):
        data.qpos[probe_position : probe_position + 3] = tool * np.array([x, 0.09, 0])
        mujoco.mj_forward(model, data)
        distance = min(
            mujoco.mj_geomDistance(model, data, geom, probe, 1.0, None) for geom in finger_geoms
        )
        if in_gap:
            # A combined mount/fin convex hull incorrectly fills this empty space.
            assert distance > MINIMUM_FINGER_GAP_CLEARANCE
        else:
            assert distance < 0  # The actual fin must still collide.
    engine.close()
