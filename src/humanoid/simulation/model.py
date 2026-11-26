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
from humanoid.types.robot import GeomContactConfig
from humanoid.types.simulation import MujocoSimulationConfig, SimulationScene

ROOT_X_JOINT = "simulation_root_x"
ROOT_Y_JOINT = "simulation_root_y"
ROOT_YAW_JOINT = "simulation_root_yaw"
ROOT_JOINT_NAMES = (ROOT_X_JOINT, ROOT_Y_JOINT, ROOT_YAW_JOINT)

_PACKAGE_URI_PATTERN = re.compile(r"package://([^/]+)/([^\"']+)")
ROBOT_CONTACT_TYPE = 1
_EXTERNAL_CONTACT_TYPE = 2
_FLOOR_CONTACT_TYPE = 4


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

    resolved_xml = _resolve_package_uris(robot.urdf_path.read_text(), robot.urdf_path)
    xml_root = ET.fromstring(resolved_xml)
    root_link_name = _find_root_link_name(xml_root)
    # URDF imports discard visuals by default. Retain them for RGB/depth rendering.
    extension = xml_root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(xml_root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.set("discardvisual", "false")
    compiler.attrib.setdefault("boundmass", str(config.minimum_body_mass))
    compiler.attrib.setdefault("boundinertia", str(config.minimum_body_inertia))
    spec = mujoco.MjSpec.from_string(ET.tostring(xml_root, encoding="unicode"))

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

    _configure_robot_contacts(spec, robot.config.geom_contacts)

    for mount in robot.config.cameras.values():
        camera = mount.simulation
        if not config.enable_cameras or camera is None:
            continue
        frames = (camera.rgb_frame, camera.depth_frame)
        bodies = [spec.body(frame) for frame in frames]
        if any(body is not None for body in bodies):
            if any(body is None for body in bodies):
                raise ValueError("Simulation camera requires both URDF optical frames.")
            # Keep fixed optical links: static-body fusion can lose the local
            # transforms of cameras added after the URDF import.
            spec.compiler.fusestatic = False
            for frame, body, fov in zip(
                frames, bodies, (camera.rgb_fov_degrees, camera.depth_fov_degrees), strict=True
            ):
                assert body is not None
                # Optical +Z forward/+Y down -> OpenGL -Z forward/+Y up.
                # At unit focal length the sensor spans twice each half-angle's tangent.
                body.add_camera(
                    name=frame,
                    quat=[0, 1, 0, 0],
                    resolution=[camera.width, camera.height],
                    sensor_size=2 * np.tan(np.deg2rad(fov) / 2),
                    focal_length=[1, 1],
                )
            spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, camera.width)
            spec.visual.global_.offheight = max(spec.visual.global_.offheight, camera.height)

    if robot.config.base is not None:
        _add_ideal_planar_root(spec, root_link_name, robot, config)
    _add_robot_actuators(spec, robot, config, xml_root)
    _add_mimic_constraints(spec, xml_root)
    if config.scene is not SimulationScene.EMPTY:
        scene_file = Path(__file__).with_name("scenes") / f"{config.scene.value}.xml"
        scene = mujoco.MjSpec.from_file(str(scene_file))
        spec.attach(scene, prefix="", frame=spec.worldbody.add_frame())
    return spec


def _configure_robot_contacts(
    spec: mujoco.MjSpec, geom_contacts: Mapping[str, GeomContactConfig]
) -> None:
    # Keep robot links from colliding with each other until the SRDF collision
    # pairs are translated. They can still collide with future scene objects
    # configured with the reciprocal external contact bits.
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype = ROBOT_CONTACT_TYPE
            geom.conaffinity = _EXTERNAL_CONTACT_TYPE | _FLOOR_CONTACT_TYPE
            geom.group = 3  # Hide collision proxies from camera images.

    for name, contact in geom_contacts.items():
        geom = spec.geom(name)
        if geom is None or not (geom.contype or geom.conaffinity):
            raise ValueError(f"No robot collision geometry named {name!r} for contact settings.")
        geom.priority = contact.priority
        geom.solref = contact.solref
        geom.solimp = contact.solimp
        if contact.sliding_friction is not None:
            geom.friction[0] = contact.sliding_friction
        if contact.torsional_friction is not None:
            geom.condim = 4
            geom.friction[1] = contact.torsional_friction


def _resolve_package_uris(urdf_xml: str, urdf_path: Path) -> str:
    assets_root = urdf_path.parents[2]

    def replace(match: re.Match[str]) -> str:
        package_name, relative_path = match.groups()
        resolved = assets_root / package_name / relative_path
        if not resolved.exists():
            raise FileNotFoundError(f"URDF package asset not found: {resolved}")
        return str(resolved)

    return _PACKAGE_URI_PATTERN.sub(replace, urdf_xml)


def _find_root_link_name(root: ET.Element) -> str:
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

    # The planar joint supports the chassis and the root actuators supply traction.
    # A rigid wheel mesh has no free rollers and would scrub against the floor.
    for index in robot.get_wheel_joint_indices():
        wheel = spec.joint(robot.joint_idx_to_name(index))
        for geom in wheel.parent.geoms:
            geom.conaffinity &= ~_FLOOR_CONTACT_TYPE

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
    urdf: ET.Element,
) -> None:
    for joint_name in robot.actuator_joint_names:
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"MuJoCo model has no controlled joint named {joint_name!r}.")
        drive = robot.config.joint_drives[joint_name]
        joint.armature = config.joint_armature if drive.armature is None else drive.armature
        dynamics = urdf.find(f"joint[@name='{joint_name}']/dynamics")
        if dynamics is None or "damping" not in dynamics.attrib:
            joint.damping = [config.joint_damping, 0.0, 0.0]

        actuator = spec.add_actuator()
        actuator.name = joint_name
        actuator.target = joint_name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        joint_idx = robot.joint_name_to_idx(joint_name)
        velocity_idx = robot.joint_idx_to_velocity_idx(joint_idx)
        if drive.control_mode is ActuatorControlMode.POSITION:
            actuator.set_to_position(
                config.position_kp,
                dampratio=config.position_damping_ratio,
                inheritrange=True,
            )
        else:
            actuator.set_to_velocity(config.velocity_kv)
            velocity_limit = float(robot.model.velocityLimit[velocity_idx])
            actuator.ctrllimited = True
            actuator.ctrlrange = [-velocity_limit, velocity_limit]

        effort_limit = float(robot.model.effortLimit[velocity_idx])
        if np.isfinite(effort_limit) and effort_limit > 0.0:
            actuator.forcelimited = True
            actuator.forcerange = [-effort_limit, effort_limit]


def _add_mimic_constraints(spec: mujoco.MjSpec, root: ET.Element) -> None:
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
