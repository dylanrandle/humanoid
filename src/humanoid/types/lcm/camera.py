"""Conversions for RGB camera messages shared by real and simulated sensors."""

import numpy as np

from humanoid.types.camera import CameraFrame
from humanoid.types.lcm import camera_frame_t


def camera_frame_to_lcm(frame: CameraFrame) -> camera_frame_t:
    message = camera_frame_t()
    message.timestamp = round(frame.timestamp * 1e9)
    message.name, message.frame_id = frame.name, frame.frame_id
    message.width, message.height = frame.width, frame.height
    message.intrinsics = frame.intrinsics.ravel().tolist()
    message.rgb_size = len(frame.rgb_jpeg)
    message.rgb_jpeg = frame.rgb_jpeg
    return message


def camera_frame_from_lcm(message: camera_frame_t) -> CameraFrame:
    return CameraFrame(
        timestamp=message.timestamp / 1e9,
        name=message.name,
        frame_id=message.frame_id,
        width=message.width,
        height=message.height,
        intrinsics=np.asarray(message.intrinsics).reshape(3, 3),
        rgb_jpeg=message.rgb_jpeg,
    )
