"""Wire conversions for ground-truth and estimated object states."""

import numpy as np
import pinocchio as pin

from humanoid.types.lcm import object_state_t
from humanoid.types.object_state import ObjectState


def object_state_to_lcm(state: ObjectState) -> object_state_t:
    message = object_state_t()
    message.timestamp = round(state.timestamp * 1e9)
    message.name, message.frame_id = state.name, state.frame_id
    message.position = state.pose.translation.tolist()
    q = pin.Quaternion(state.pose.rotation)
    message.quaternion = [q.w, q.x, q.y, q.z]
    message.has_bbox = state.bbox_center is not None and state.bbox_half_size is not None
    if state.bbox_center is not None and state.bbox_half_size is not None:
        message.bbox_center = state.bbox_center.tolist()
        message.bbox_half_size = state.bbox_half_size.tolist()
    message.has_velocity = state.linear_velocity is not None and state.angular_velocity is not None
    if state.linear_velocity is not None and state.angular_velocity is not None:
        message.linear_velocity = state.linear_velocity.tolist()
        message.angular_velocity = state.angular_velocity.tolist()
    return message


def object_state_from_lcm(message: object_state_t) -> ObjectState:
    return ObjectState(
        timestamp=message.timestamp / 1e9,
        name=message.name,
        frame_id=message.frame_id,
        pose=pin.SE3(
            pin.Quaternion(*message.quaternion).toRotationMatrix(), np.asarray(message.position)
        ),
        bbox_center=np.asarray(message.bbox_center) if message.has_bbox else None,
        bbox_half_size=np.asarray(message.bbox_half_size) if message.has_bbox else None,
        linear_velocity=np.asarray(message.linear_velocity) if message.has_velocity else None,
        angular_velocity=np.asarray(message.angular_velocity) if message.has_velocity else None,
    )
