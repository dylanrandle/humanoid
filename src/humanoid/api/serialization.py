"""JSON state and lossless PNG camera snapshots for agent clients."""

import base64
import io
import time
from dataclasses import asdict

import numpy as np
from PIL import Image

from humanoid.types.actuator import ActuatorEffortSource
from humanoid.types.api import ApiKey
from humanoid.types.camera import (
    CameraEncoding,
    CameraFrames,
    CameraImage,
    CameraIntrinsics,
    CameraView,
    DepthCamera,
    DepthView,
    RegisteredDepth,
)
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
            ApiKey.TIMESTAMP: frame.timestamp,
            ApiKey.AGE_SECONDS: max(0.0, captured_at - frame.timestamp),
            ApiKey.DEVICE_TIMESTAMP: frame.device_timestamp,
            ApiKey.TIMESTAMP_DOMAIN: frame.timestamp_domain,
            ApiKey.FRAME_NUMBER: frame.frame_number,
            ApiKey.FRAME_ID: frame.frame_id,
            ApiKey.SERIAL_NUMBER: frame.serial_number,
            ApiKey.ENCODING: frame.encoding,
            ApiKey.INTRINSICS: asdict(frame.intrinsics),
            ApiKey.DEPTH_SCALE: frame.depth_scale,
            ApiKey.PNG_BASE64: base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    view = DepthView(depth_view)
    reg = frames.registered if view is not DepthView.NATIVE else None
    depth = frames.depth
    if view is DepthView.ALIGNED:
        depth = reg.image if reg else None
    payload = {
        ApiKey.RGB: encode(frames.rgb),
        ApiKey.DEPTH_VIEW: view.value,
        ApiKey.DEPTH: encode(depth) if depth is not None else None,
    }
    if reg is not None:
        payload[ApiKey.REGISTERED] = {
            ApiKey.CAMERA_MODEL: reg.camera_model,
            ApiKey.RGB_FRAME_NUMBER: reg.rgb_frame_number,
            **({"image": encode(reg.image)} if view is DepthView.BOTH else {}),
        }
    return payload


def decode_frames(payload: dict, received_after: float) -> CameraView:
    def decode(frame: dict) -> CameraImage:
        encoding = CameraEncoding(frame[ApiKey.ENCODING])
        with Image.open(
            io.BytesIO(base64.b64decode(frame[ApiKey.PNG_BASE64], validate=True))
        ) as image:
            data = np.array(image, dtype=np.uint8 if encoding is CameraEncoding.RGB8 else np.uint16)
        calibration = dict(frame[ApiKey.INTRINSICS])
        calibration[ApiKey.DISTORTION_COEFFICIENTS] = tuple(
            calibration[ApiKey.DISTORTION_COEFFICIENTS]
        )
        return CameraImage(
            timestamp=received_after - frame[ApiKey.AGE_SECONDS],
            device_timestamp=frame[ApiKey.DEVICE_TIMESTAMP],
            timestamp_domain=frame[ApiKey.TIMESTAMP_DOMAIN],
            frame_number=frame[ApiKey.FRAME_NUMBER],
            frame_id=frame[ApiKey.FRAME_ID],
            serial_number=frame[ApiKey.SERIAL_NUMBER],
            encoding=encoding,
            data=data,
            intrinsics=CameraIntrinsics(**calibration),
            depth_scale=frame[ApiKey.DEPTH_SCALE],
        )

    reg = payload.get(ApiKey.REGISTERED)
    view = DepthView(payload[ApiKey.DEPTH_VIEW])
    rgb = decode(payload[ApiKey.RGB])
    depth = decode(payload[ApiKey.DEPTH]) if payload[ApiKey.DEPTH] is not None else None
    if view is not DepthView.ALIGNED:
        if depth is None or depth.encoding is not CameraEncoding.DEPTH16:
            raise ValueError("Expected a native depth image.")
        if view is DepthView.NATIVE and reg is not None:
            raise ValueError("Unexpected registration metadata in a native-only view.")
        registered = (
            None
            if reg is None
            else RegisteredDepth(
                camera_model=DepthCamera(reg[ApiKey.CAMERA_MODEL]),
                rgb_frame_number=reg[ApiKey.RGB_FRAME_NUMBER],
                image=decode(reg["image"]),
            )
        )
        return CameraFrames(rgb=rgb, depth=depth, registered=registered)
    if (reg is None) != (depth is None):
        raise ValueError("Aligned depth and its registration metadata must be supplied together.")
    registered = None
    if reg is not None:
        assert depth is not None
        registered = RegisteredDepth(
            camera_model=DepthCamera(reg[ApiKey.CAMERA_MODEL]),
            rgb_frame_number=reg[ApiKey.RGB_FRAME_NUMBER],
            image=depth,
        )
    return CameraView(rgb=rgb, depth=depth, registered=registered)
