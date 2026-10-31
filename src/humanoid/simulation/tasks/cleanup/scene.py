"""Physical floor, loose toys, open bin, and the robot's RGB cameras."""

import math

import mujoco
import numpy as np

from humanoid.simulation.camera import add_camera
from humanoid.types.tasks.cleanup import CleanupSceneConfig, SceneObjectConfig


def add_cleanup_scene(spec: mujoco.MjSpec, config: CleanupSceneConfig) -> None:
    """Compose a scene before compilation; only physics moves the free objects."""
    # Soft-contact regularization otherwise creeps even inside the static
    # friction cone. See MuJoCo's modeling guide, "Preventing slip".
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = config.friction_impedance_ratio
    spec.option.noslip_iterations = config.noslip_iterations
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, config.floor_height],
        size=[2, 2, 0.05],
        rgba=[0.42, 0.43, 0.45, 1],
        contype=2,
        conaffinity=3,
        friction=[0.8, 0.005, 0.0001],
    )
    spec.worldbody.add_light(pos=[0, 0.3, 1.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    for name, camera in config.cameras.items():
        add_camera(spec, name, camera)
    _add_bin(spec, config)
    for item in config.objects:
        _add_object(spec, item, config.floor_height)


def _add_bin(spec: mujoco.MjSpec, config: CleanupSceneConfig) -> None:
    x, y = config.bin_half_size
    t = config.wall_thickness
    h = config.bin_height
    body = spec.worldbody.add_body(
        name="cleanup_bin",
        pos=[*config.bin_position, config.floor_height],
    )
    # Five separate boxes leave the top and interior physically open.
    parts = (
        ([0, 0, t / 2], [x, y, t / 2]),
        ([-x + t / 2, 0, h / 2], [t / 2, y, h / 2]),
        ([x - t / 2, 0, h / 2], [t / 2, y, h / 2]),
        ([0, -y + t / 2, h / 2], [x - t, t / 2, h / 2]),
        ([0, y - t / 2, h / 2], [x - t, t / 2, h / 2]),
    )
    for index, (pos, size) in enumerate(parts):
        body.add_geom(
            name=f"bin_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=[0.04, 0.18, 0.85, 1],
            contype=2,
            conaffinity=3,
            friction=[0.8, 0.005, 0.0001],
        )


def _add_object(spec: mujoco.MjSpec, item: SceneObjectConfig, floor: float) -> None:
    half = np.asarray(item.size) / 2
    body = spec.worldbody.add_body(
        name=item.name,
        pos=[*item.position, floor + half[2] + 0.002],
        quat=[math.cos(item.yaw / 2), 0, 0, math.sin(item.yaw / 2)],
    )
    body.add_freejoint(name=f"{item.name}_free")
    geom = body.add_geom(
        name=f"{item.name}_shape",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        mass=item.mass,
        rgba=item.color,
        contype=2,
        conaffinity=3,
        condim=4,
        friction=[1.0, 0.01, 0.0001],
        solref=[0.008, 1],
    )
    if item.kind == "car":
        # A rigid toy car: graspable chassis, cabin, and four rubber wheels.
        geom.size = [half[0], half[1] * 0.8, half[2] * 0.6]
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-half[0] * 0.15, 0, half[2] * 0.65],
            size=[half[0] * 0.48, half[1] * 0.68, half[2] * 0.35],
            mass=0.005,
            rgba=item.color,
            contype=2,
            conaffinity=3,
        )
        for side in (-1, 1):
            for end in (-1, 1):
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[end * half[0] * 0.65, side * half[1] * 0.9, -half[2] * 0.4],
                    size=[half[2] * 0.6, half[1] * 0.15, 0],
                    quat=[math.sqrt(0.5), math.sqrt(0.5), 0, 0],
                    mass=0.002,
                    rgba=[0.04, 0.04, 0.04, 1],
                    contype=2,
                    conaffinity=3,
                    friction=[1, 0.005, 0.0001],
                )
