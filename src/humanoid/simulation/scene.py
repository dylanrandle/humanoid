"""Compose the canonical robot model into a named MuJoCo scene."""

from __future__ import annotations

import mujoco

from humanoid.config.simulation import (
    DEFAULT_MUJOCO_SIMULATION_CONFIG,
    FLOOR_AND_CUBE_SCENE_CONFIG,
)
from humanoid.robots.base import Robot
from humanoid.simulation.model import (
    ROBOT_CONTACT_BIT,
    TASK_OBJECT_CONTACT_BIT,
    build_mujoco_spec,
)
from humanoid.types.simulation import (
    FloorAndCubeSceneConfig,
    MujocoScene,
    MujocoSimulationConfig,
)

FLOOR_GEOM_NAME = "scene_floor"
CUBE_BODY_NAME = "scene_cube"
CUBE_JOINT_NAME = "scene_cube_freejoint"
CUBE_GEOM_NAME = "scene_cube_geom"

_FLOOR_CONTACT_BIT = 4
_FLOOR_RGBA = (0.82, 0.84, 0.87, 1.0)
_CUBE_RGBA = (0.9, 0.25, 0.1, 1.0)


def build_mujoco_scene(
    robot: Robot,
    scene: MujocoScene,
    simulation_config: MujocoSimulationConfig = DEFAULT_MUJOCO_SIMULATION_CONFIG,
    floor_and_cube_config: FloorAndCubeSceneConfig = FLOOR_AND_CUBE_SCENE_CONFIG,
) -> mujoco.MjSpec:
    """Build a named scene as an uncompiled ``MjSpec``."""

    spec = build_mujoco_spec(robot, simulation_config)
    if scene is MujocoScene.FLOOR_AND_CUBE:
        _add_floor(spec, floor_and_cube_config)
        _add_cube(spec, floor_and_cube_config)
    return spec


def _add_floor(spec: mujoco.MjSpec, config: FloorAndCubeSceneConfig) -> None:
    spec.worldbody.add_geom(
        name=FLOOR_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[config.floor_half_extent, config.floor_half_extent, 0.1],
        contype=_FLOOR_CONTACT_BIT,
        conaffinity=TASK_OBJECT_CONTACT_BIT,
        rgba=_FLOOR_RGBA,
    )


def _add_cube(spec: mujoco.MjSpec, config: FloorAndCubeSceneConfig) -> None:
    half_extent = config.cube_edge_length / 2.0
    cube = spec.worldbody.add_body(
        name=CUBE_BODY_NAME,
        pos=[config.cube_x, config.cube_y, half_extent],
    )
    cube.add_freejoint(name=CUBE_JOINT_NAME)
    cube.add_geom(
        name=CUBE_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half_extent, half_extent, half_extent],
        mass=config.cube_mass,
        contype=TASK_OBJECT_CONTACT_BIT,
        conaffinity=ROBOT_CONTACT_BIT | TASK_OBJECT_CONTACT_BIT | _FLOOR_CONTACT_BIT,
        rgba=_CUBE_RGBA,
    )
