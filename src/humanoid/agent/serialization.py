"""JSON state and lossless PNG camera snapshots for agent clients."""

import base64
import io
import time
from dataclasses import asdict

import numpy as np
from PIL import Image

from humanoid.types.actuator import ActuatorEffortSource
from humanoid.types.camera import (
    CameraEncoding,
    CameraFrames,
    CameraImage,
    CameraIntrinsics,
    CameraView,
    DepthView,
    RegisteredDepth,
)
from humanoid.types.depth import DepthCamera
from humanoid.types.http import HttpKey
from humanoid.types.robot import RobotState


def encode_state(state: RobotState) -> dict:
    payload = asdict(state)
    for name in ("joint_positions", "joint_velocities", "joint_efforts", "actuator_temperatures"):
        values = payload[name]
        if values is not None:
            payload[name] = [float(value) if np.isfinite(value) else None for value in values]
    payload["age_seconds"] = max(0.0, time.perf_counter() - state.timestamp)
    return payload


def decode_state(payload: dict, received_after: float) -> RobotState:
    return RobotState(
        timestamp=received_after - payload["age_seconds"],
        joint_positions=np.asarray(payload["joint_positions"], dtype=float),
        joint_velocities=np.asarray(payload["joint_velocities"], dtype=float),
        joint_efforts=None
        if payload["joint_efforts"] is None
        else np.asarray(payload["joint_efforts"], dtype=float),
        actuator_temperatures=np.asarray(payload["actuator_temperatures"], dtype=float),
        effort_source=None
        if payload["effort_source"] is None
        else ActuatorEffortSource(payload["effort_source"]),
    )


def encode_frames(frames: CameraFrames, *, depth_view: DepthView | str = DepthView.ALIGNED) -> dict:
    captured_at = time.perf_counter()

    def encode(frame: CameraImage) -> dict:
        buffer = io.BytesIO()
        Image.fromarray(frame.data).save(buffer, format="PNG")
        return {
            HttpKey.TIMESTAMP: frame.timestamp,
            HttpKey.AGE_SECONDS: max(0.0, captured_at - frame.timestamp),
            HttpKey.DEVICE_TIMESTAMP: frame.device_timestamp,
            HttpKey.TIMESTAMP_DOMAIN: frame.timestamp_domain,
            HttpKey.FRAME_NUMBER: frame.frame_number,
            HttpKey.FRAME_ID: frame.frame_id,
            HttpKey.SERIAL_NUMBER: frame.serial_number,
            HttpKey.ENCODING: frame.encoding,
            HttpKey.INTRINSICS: asdict(frame.intrinsics),
            HttpKey.DEPTH_SCALE: frame.depth_scale,
            HttpKey.PNG_BASE64: base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    view = DepthView(depth_view)
    reg = frames.registered if view is DepthView.ALIGNED else None
    depth = frames.depth
    if view is DepthView.ALIGNED:
        depth = reg.image if reg else None
    payload = {
        HttpKey.RGB: encode(frames.rgb),
        HttpKey.DEPTH_VIEW: view.value,
        HttpKey.DEPTH: encode(depth) if depth is not None else None,
    }
    if reg is not None:
        payload[HttpKey.REGISTERED] = {
            HttpKey.CAMERA_MODEL: reg.camera_model,
            HttpKey.RGB_FRAME_NUMBER: reg.rgb_frame_number,
        }
    return payload


def decode_frames(payload: dict, received_after: float) -> CameraView:
    def decode(frame: dict) -> CameraImage:
        encoding = CameraEncoding(frame[HttpKey.ENCODING])
        with Image.open(
            io.BytesIO(base64.b64decode(frame[HttpKey.PNG_BASE64], validate=True))
        ) as image:
            data = np.array(image, dtype=np.uint8 if encoding is CameraEncoding.RGB8 else np.uint16)
        calibration = dict(frame[HttpKey.INTRINSICS])
        calibration[HttpKey.DISTORTION_COEFFICIENTS] = tuple(
            calibration[HttpKey.DISTORTION_COEFFICIENTS]
        )
        return CameraImage(
            timestamp=received_after - frame[HttpKey.AGE_SECONDS],
            device_timestamp=frame[HttpKey.DEVICE_TIMESTAMP],
            timestamp_domain=frame[HttpKey.TIMESTAMP_DOMAIN],
            frame_number=frame[HttpKey.FRAME_NUMBER],
            frame_id=frame[HttpKey.FRAME_ID],
            serial_number=frame[HttpKey.SERIAL_NUMBER],
            encoding=encoding,
            data=data,
            intrinsics=CameraIntrinsics(**calibration),
            depth_scale=frame[HttpKey.DEPTH_SCALE],
        )

    reg = payload.get(HttpKey.REGISTERED)
    view = DepthView(payload[HttpKey.DEPTH_VIEW])
    rgb = decode(payload[HttpKey.RGB])
    depth = decode(payload[HttpKey.DEPTH]) if payload[HttpKey.DEPTH] is not None else None
    if view is DepthView.NATIVE:
        if reg is not None or depth is None:
            raise ValueError("Native depth requires a depth image without registration metadata.")
        return CameraFrames(rgb=rgb, depth=depth)
    if (reg is None) != (depth is None):
        raise ValueError("Aligned depth and its registration metadata must be supplied together.")
    registered = None
    if reg is not None:
        assert depth is not None
        registered = RegisteredDepth(
            camera_model=DepthCamera(reg[HttpKey.CAMERA_MODEL]),
            rgb_frame_number=reg[HttpKey.RGB_FRAME_NUMBER],
            image=depth,
        )
    return CameraView(rgb=rgb, depth=depth, registered=registered)
