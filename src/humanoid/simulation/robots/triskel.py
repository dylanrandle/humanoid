"""Contact geometry for Triskel's concave parallel fingers.

MuJoCo convexifies each mesh. A single hull across a finger's carriage and tip
fills its concavity and blocks insertion. These boxes approximate the measured
carriage, stem and contact pad while the original mesh remains the visual asset.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from humanoid.types.robot import GripperSimulationConfig


def configure_triskel_gripper(
    spec: mujoco.MjSpec, urdf_xml: str, config: GripperSimulationConfig | None
) -> None:
    root = ET.fromstring(urdf_xml)
    for link, joint, sign, center_x in (
        ("finger_1_link", "gripper_3", 1, 0.029),
        ("finger_2_link", "gripper_2", -1, -0.026),
    ):
        body = spec.body(link)
        origin = root.find(f"joint[@name='{joint}']/origin")
        if body is None or origin is None:
            raise ValueError(f"Missing Triskel finger geometry: {link}")
        offset = np.fromstring(origin.attrib["xyz"], sep=" ")
        for geom in body.geoms:
            geom.contype = 0
            geom.conaffinity = 0
        parts = (
            ([sign * 0.011, 0.054, 0], [0.027, 0.009, 0.031]),
            ([center_x, 0.073, 0], [0.0023, 0.010, 0.019]),
            ([center_x, 0.091, 0], [0.0023, 0.012, 0.013]),
        )
        for index, (center, half_size) in enumerate(parts):
            geom = body.add_geom(
                name=f"{link}_contact_{index}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=np.asarray(center) - offset,
                size=half_size,
                mass=0,
                rgba=[0, 0, 0, 0],
                contype=1,
                conaffinity=2,
            )
            if config is not None:
                geom.friction = config.friction
                geom.condim = config.contact_dimension
                geom.solref = config.contact_solref
                geom.solimp = config.contact_solimp
    if config is None:
        return
    if config.motor_torque_limit is not None:
        actuator = spec.actuator("gripper_1")
        if actuator is None:
            raise ValueError("Missing Triskel gripper actuator")
        actuator.forcelimited = True
        actuator.forcerange = [-config.motor_torque_limit, config.motor_torque_limit]
    for equality in spec.equalities:
        if equality.name.startswith(("gripper_2_", "gripper_3_")):
            equality.solref = config.coupling_solref
            equality.solimp = config.coupling_solimp
