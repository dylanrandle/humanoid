"""Explicit name-based bindings into a compiled MuJoCo robot model."""

from dataclasses import dataclass
from typing import cast

import mujoco

from humanoid.robots.base import Robot
from humanoid.simulation.model import ROOT_JOINT_NAMES
from humanoid.types.actuator import ActuatorControlMode


@dataclass(frozen=True)
class MujocoJointBinding:
    name: str
    control_mode: ActuatorControlMode
    joint_id: int
    actuator_id: int
    qpos_address: int
    qvel_address: int


@dataclass(frozen=True)
class MujocoRootBinding:
    joint_ids: tuple[int, int, int]
    actuator_ids: tuple[int, int, int]
    qpos_addresses: tuple[int, int, int]
    qvel_addresses: tuple[int, int, int]


@dataclass(frozen=True)
class MujocoRobotBinding:
    joints: tuple[MujocoJointBinding, ...]
    root: MujocoRootBinding | None


def resolve_mujoco_robot_binding(
    model: mujoco.MjModel,
    robot: Robot,
) -> MujocoRobotBinding:
    """Resolve all robot-controlled indices without relying on model ordering."""

    joints = tuple(_resolve_joint(model, robot, name) for name in robot.actuator_joint_names)
    root = _resolve_root(model) if robot.config.base is not None else None
    return MujocoRobotBinding(joints=joints, root=root)


def _resolve_joint(model: mujoco.MjModel, robot: Robot, name: str) -> MujocoJointBinding:
    joint_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    actuator_id = _required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    transmission_joint_id = int(model.actuator_trnid[actuator_id, 0])
    if transmission_joint_id != joint_id:
        raise ValueError(f"MuJoCo actuator {name!r} does not target its matching joint.")
    return MujocoJointBinding(
        name=name,
        control_mode=robot.config.actuator_control_modes[name],
        joint_id=joint_id,
        actuator_id=actuator_id,
        qpos_address=int(model.jnt_qposadr[joint_id]),
        qvel_address=int(model.jnt_dofadr[joint_id]),
    )


def _resolve_root(model: mujoco.MjModel) -> MujocoRootBinding:
    joint_ids = cast(
        tuple[int, int, int],
        tuple(_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ROOT_JOINT_NAMES),
    )
    actuator_ids = cast(
        tuple[int, int, int],
        tuple(_required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ROOT_JOINT_NAMES),
    )
    return MujocoRootBinding(
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        qpos_addresses=cast(
            tuple[int, int, int],
            tuple(int(model.jnt_qposadr[index]) for index in joint_ids),
        ),
        qvel_addresses=cast(
            tuple[int, int, int],
            tuple(int(model.jnt_dofadr[index]) for index in joint_ids),
        ),
    )


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo model has no {object_type.name.lower()} named {name!r}.")
    return object_id
