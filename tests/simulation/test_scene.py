"""Tests for the default floor-and-cube MuJoCo scene."""

import mujoco
import pytest

from humanoid.config.robot.so101 import SO101_CONFIG
from humanoid.robots.base import Robot
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.model import ROBOT_CONTACT_BIT, TASK_OBJECT_CONTACT_BIT
from humanoid.simulation.scene import (
    CUBE_BODY_NAME,
    CUBE_GEOM_NAME,
    CUBE_JOINT_NAME,
    FLOOR_GEOM_NAME,
    build_mujoco_scene,
)
from humanoid.types.simulation import FloorAndCubeSceneConfig, MujocoScene


def test_scene_contains_a_floor_and_free_cube_with_configured_properties():
    config = FloorAndCubeSceneConfig(
        floor_half_extent=3.0,
        cube_edge_length=0.06,
        cube_mass=0.2,
        cube_x=0.4,
        cube_y=-0.1,
    )
    model = build_mujoco_scene(
        Robot(SO101_CONFIG),
        MujocoScene.FLOOR_AND_CUBE,
        floor_and_cube_config=config,
    ).compile()

    floor_geom_id = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    cube_body_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME)
    cube_geom_id = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM_NAME)
    cube_joint_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT_NAME)
    cube_qpos_address = model.jnt_qposadr[cube_joint_id]

    assert model.geom_type[floor_geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
    assert model.geom_size[floor_geom_id, :2] == pytest.approx([3.0, 3.0])
    assert model.geom_type[cube_geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_size[cube_geom_id] == pytest.approx([0.03, 0.03, 0.03])
    assert model.body_mass[cube_body_id] == pytest.approx(0.2)
    assert model.jnt_type[cube_joint_id] == mujoco.mjtJoint.mjJNT_FREE
    assert model.qpos0[cube_qpos_address : cube_qpos_address + 3] == pytest.approx(
        [0.4, -0.1, 0.03]
    )


def test_empty_scene_contains_only_the_robot_model():
    model = build_mujoco_scene(Robot(SO101_CONFIG), MujocoScene.EMPTY).compile()

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME) == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME) == -1


def test_cube_collides_with_the_floor_and_robot_but_floor_ignores_robot():
    model = build_mujoco_scene(Robot(SO101_CONFIG), MujocoScene.FLOOR_AND_CUBE).compile()
    floor_geom_id = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    cube_geom_id = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM_NAME)
    robot_geom_id = next(
        geom_id for geom_id in range(model.ngeom) if geom_id not in {floor_geom_id, cube_geom_id}
    )

    assert model.geom_contype[robot_geom_id] & ROBOT_CONTACT_BIT
    assert model.geom_conaffinity[robot_geom_id] & TASK_OBJECT_CONTACT_BIT
    assert model.geom_contype[cube_geom_id] & TASK_OBJECT_CONTACT_BIT
    assert model.geom_conaffinity[cube_geom_id] & ROBOT_CONTACT_BIT
    assert model.geom_contype[floor_geom_id] & model.geom_conaffinity[cube_geom_id]
    assert not (model.geom_contype[floor_geom_id] & model.geom_conaffinity[robot_geom_id])
    assert not (model.geom_contype[robot_geom_id] & model.geom_conaffinity[floor_geom_id])


def test_cube_settles_on_the_floor():
    scene_config = FloorAndCubeSceneConfig(cube_x=1.0, cube_y=1.0)
    engine = NativeMujocoEngine(
        SO101_CONFIG,
        scene=MujocoScene.FLOOR_AND_CUBE,
        floor_and_cube_config=scene_config,
    )
    cube_body_id = _required_id(engine.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME)

    engine.step(1000)

    assert engine.data.xpos[cube_body_id, 2] == pytest.approx(
        scene_config.cube_edge_length / 2.0,
        abs=2e-4,
    )


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    assert object_id >= 0
    return object_id
