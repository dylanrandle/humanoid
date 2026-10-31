"""Privileged rigid-body geometry and velocity queries for simulation tasks."""

import mujoco
import numpy as np
import pinocchio as pin

from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.object_state import ObjectState


def body_bounding_box(engine: NativeMujocoEngine, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the box/cylinder collision shapes in the body's local axes."""
    model = engine.model
    body = model.body(name).id
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for geom_id in np.flatnonzero(model.geom_bodyid == body):
        geom = model.geom(int(geom_id))
        if not geom.contype:
            continue
        size = geom.size.copy()
        if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            size = np.array([size[0], size[0], size[1]])
        elif geom.type != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"Unsupported bounding-box geometry on {name}: {geom.type}")
        quat = pin.Quaternion(*geom.quat)
        extent = np.abs(quat.toRotationMatrix()) @ size
        lower = np.minimum(lower, geom.pos - extent)
        upper = np.maximum(upper, geom.pos + extent)
    if not np.isfinite([lower, upper]).all():
        raise ValueError(f"Body has no bounded collision geometry: {name}")
    return (lower + upper) / 2, (upper - lower) / 2


def read_object(
    engine: NativeMujocoEngine, name: str, bounds: tuple[np.ndarray, np.ndarray]
) -> ObjectState:
    body = engine.data.body(name)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        engine.model, engine.data, mujoco.mjtObj.mjOBJ_BODY, body.id, velocity, 0
    )
    return ObjectState(
        timestamp=engine.data.time,
        name=name,
        frame_id="world",
        pose=pin.SE3(body.xmat.reshape(3, 3).copy(), body.xpos.copy()),
        bbox_center=bounds[0],
        bbox_half_size=bounds[1],
        linear_velocity=velocity[3:],
        angular_velocity=velocity[:3],
    )
