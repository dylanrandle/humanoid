"""Build the canonical MuJoCo model specification from Humanoid robot assets."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np

from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.robots.base import Robot
from humanoid.types.actuator import ActuatorControlMode
from humanoid.types.robot import MeshContactConfig
from humanoid.types.simulation import MujocoSimulationConfig, SimulationScene

ROOT_X_JOINT = "simulation_root_x"
ROOT_Y_JOINT = "simulation_root_y"
ROOT_YAW_JOINT = "simulation_root_yaw"
ROOT_JOINT_NAMES = (ROOT_X_JOINT, ROOT_Y_JOINT, ROOT_YAW_JOINT)

_PACKAGE_URI_PATTERN = re.compile(r"package://([^/]+)/([^\"']+)")
_ROBOT_CONTACT_TYPE = 1
_EXTERNAL_CONTACT_TYPE = 2


def build_mujoco_spec(
    robot: Robot,
    config: MujocoSimulationConfig = DEFAULT_MUJOCO_SIMULATION_CONFIG,
) -> mujoco.MjSpec:
    """Build one reusable ``MjSpec`` for native simulation and future training.

    The URDF remains the kinematic source used by Pinocchio. This builder imports
    that same asset into MuJoCo, adds the actuator and ideal-planar-base semantics
    owned by Humanoid, and returns the uncompiled specification so other runtimes
    can compose it into their own scenes.
    """

    urdf_xml = robot.urdf_path.read_text()
    root_link_name = _find_root_link_name(urdf_xml)
    resolved_xml = _resolve_package_uris(urdf_xml, robot.urdf_path)
    # URDF imports discard visuals by default. Retain them for RGB/depth rendering.
    xml_root = ET.fromstring(resolved_xml)
    extension = xml_root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(xml_root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.set("discardvisual", "false")
    spec = mujoco.MjSpec.from_string(ET.tostring(xml_root, encoding="unicode"))

    spec.compiler.boundmass = config.minimum_body_mass
    spec.compiler.boundinertia = config.minimum_body_inertia
    spec.compiler.balanceinertia = True
    spec.option.timestep = config.physics_timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # Resolve residual sliding at static friction contacts (notably a held toy).
    spec.option.noslip_iterations = 4

    # Some generated OBJ/STL meshes are open shells. MuJoCo can still compute a
    # physically useful inertia from their surface instead of rejecting them as
    # zero-volume solids.
    for mesh in spec.meshes:
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL

    _configure_robot_contacts(spec, robot.config.mesh_contacts)

    camera = config.camera
    if camera is not None:
        frames = (camera.rgb_frame, camera.depth_frame)
        bodies = [spec.body(frame) for frame in frames]
        if any(body is not None for body in bodies):
            if any(body is None for body in bodies):
                raise ValueError("Simulation camera requires both URDF optical frames.")
            # Keep fixed optical links: static-body fusion can lose the local
            # transforms of cameras added after the URDF import.
            spec.compiler.fusestatic = False
            for frame, body, fov in zip(
                frames, bodies, (camera.rgb_vertical_fov, camera.depth_vertical_fov), strict=True
            ):
                assert body is not None
                # Optical +Z forward/+Y down -> OpenGL -Z forward/+Y up.
                body.add_camera(name=frame, quat=[0, 1, 0, 0], fovy=fov)
            spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, camera.width)
            spec.visual.global_.offheight = max(spec.visual.global_.offheight, camera.height)

    if robot.config.base is not None:
        _add_ideal_planar_root(spec, root_link_name, robot, config)
    _add_robot_actuators(spec, robot, config)
    _add_mimic_constraints(spec, urdf_xml)
    if config.scene is SimulationScene.TOY_CLEANUP:
        scene_file = Path(__file__).with_name("scenes") / "toy_cleanup.xml"
        scene = mujoco.MjSpec.from_file(str(scene_file))
        spec.attach(scene, prefix="", frame=spec.worldbody.add_frame())
    return spec


def _configure_robot_contacts(
    spec: mujoco.MjSpec, mesh_contacts: Mapping[str, MeshContactConfig]
) -> None:
    # Keep robot links from colliding with each other until the SRDF collision
    # pairs are translated. They can still collide with future scene objects
    # configured with the reciprocal external contact bits.
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype = _ROBOT_CONTACT_TYPE
            geom.conaffinity = _EXTERNAL_CONTACT_TYPE
            geom.group = 3  # Hide collision proxies from camera images.

    for name, contact in mesh_contacts.items():
        geom = spec.geom(name)
        if geom is not None:
            spec.mesh(geom.meshname).maxhullvert = contact.max_hull_vertices
            geom.priority = contact.priority
            geom.solref = contact.solref
            geom.solimp = contact.solimp


def _resolve_package_uris(urdf_xml: str, urdf_path: Path) -> str:
    assets_root = urdf_path.parents[2]

    def replace(match: re.Match[str]) -> str:
        package_name, relative_path = match.groups()
        resolved = assets_root / package_name / relative_path
        if not resolved.exists():
            raise FileNotFoundError(f"URDF package asset not found: {resolved}")
        return str(resolved)

    return _PACKAGE_URI_PATTERN.sub(replace, urdf_xml)


def _find_root_link_name(urdf_xml: str) -> str:
    root = ET.fromstring(urdf_xml)
    link_names = {link.attrib["name"] for link in root.findall("link")}
    child_names = {
        child.attrib["link"]
        for joint in root.findall("joint")
        if (child := joint.find("child")) is not None
    }
    candidates = link_names - child_names
    if len(candidates) != 1:
        names = ", ".join(sorted(candidates)) or "none"
        raise ValueError(f"URDF must have exactly one root link; found {names}.")
    return candidates.pop()


def _add_ideal_planar_root(
    spec: mujoco.MjSpec,
    root_link_name: str,
    robot: Robot,
    config: MujocoSimulationConfig,
) -> None:
    root_body = spec.body(root_link_name)
    if root_body is None:
        raise ValueError(f"MuJoCo model has no root body named {root_link_name!r}.")

    damping = np.array([config.joint_damping, 0.0, 0.0])
    root_body.add_joint(
        name=ROOT_X_JOINT,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1.0, 0.0, 0.0],
        armature=config.joint_armature,
        damping=damping,
    )
    root_body.add_joint(
        name=ROOT_Y_JOINT,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0.0, 1.0, 0.0],
        armature=config.joint_armature,
        damping=damping,
    )
    root_body.add_joint(
        name=ROOT_YAW_JOINT,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 0.0, 1.0],
        armature=config.joint_armature,
        damping=damping,
    )

    base_config = robot.config.base
    if base_config is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError("Ideal planar root requires mobile-base configuration.")
    control_limits = (
        base_config.velocity_limits.linear,
        base_config.velocity_limits.linear,
        base_config.velocity_limits.angular,
    )
    for joint_name, control_limit in zip(ROOT_JOINT_NAMES, control_limits, strict=True):
        actuator = spec.add_actuator()
        actuator.name = joint_name
        actuator.target = joint_name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.set_to_velocity(config.root_velocity_kv)
        actuator.ctrllimited = True
        actuator.ctrlrange = [-control_limit, control_limit]
        actuator.forcelimited = True
        actuator.forcerange = [-config.root_force_limit, config.root_force_limit]


def _add_robot_actuators(
    spec: mujoco.MjSpec,
    robot: Robot,
    config: MujocoSimulationConfig,
) -> None:
    damping = np.array([config.joint_damping, 0.0, 0.0])
    for joint_name in robot.actuator_joint_names:
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"MuJoCo model has no controlled joint named {joint_name!r}.")
        joint.armature = config.joint_armature
        joint.damping = damping

        actuator = spec.add_actuator()
        actuator.name = joint_name
        actuator.target = joint_name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        mode = robot.config.actuator_control_modes[joint_name]
        if mode is ActuatorControlMode.POSITION:
            actuator.set_to_position(
                config.position_kp,
                dampratio=config.position_damping_ratio,
                inheritrange=True,
            )
        else:
            actuator.set_to_velocity(config.velocity_kv)
            joint_idx = robot.joint_name_to_idx(joint_name)
            velocity_idx = robot.joint_idx_to_velocity_idx(joint_idx)
            velocity_limit = float(robot.model.velocityLimit[velocity_idx])
            actuator.ctrllimited = True
            actuator.ctrlrange = [-velocity_limit, velocity_limit]

        joint_idx = robot.joint_name_to_idx(joint_name)
        effort_limit = float(robot.model.effortLimit[robot.joint_idx_to_velocity_idx(joint_idx)])
        if np.isfinite(effort_limit) and effort_limit > 0.0:
            actuator.forcelimited = True
            actuator.forcerange = [-effort_limit, effort_limit]


def _add_mimic_constraints(spec: mujoco.MjSpec, urdf_xml: str) -> None:
    root = ET.fromstring(urdf_xml)
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        joint_name = joint.attrib["name"]
        source_joint = mimic.attrib["joint"]
        multiplier = float(mimic.attrib.get("multiplier", "1"))
        offset = float(mimic.attrib.get("offset", "0"))
        equality = spec.add_equality(
            name=f"{joint_name}_mimics_{source_joint}",
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=joint_name,
            name2=source_joint,
            data=[offset, multiplier, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            active=True,
        )
        # A geared rack must track its driver even under opposing contact loads.
        if joint.attrib["type"] == "prismatic" and (
            spec.joint(source_joint).type == mujoco.mjtJoint.mjJNT_HINGE
        ):
            equality.solref = [0.01, 1]
            equality.solimp = [0.99, 0.999, 0.001, 0.5, 2]
