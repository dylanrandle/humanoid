"""JSON state and lossless PNG camera snapshots for agent clients."""

import base64
import io
import time
from dataclasses import asdict

import numpy as np
from PIL import Image

from humanoid.types.actuator import ActuatorEffortSource
from humanoid.types.camera import CameraEncoding, CameraFrames, CameraImage, CameraIntrinsics
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


def encode_frames(frames: CameraFrames) -> dict:
    captured_at = time.perf_counter()

    def encode(frame: CameraImage) -> dict:
        buffer = io.BytesIO()
        Image.fromarray(frame.data).save(buffer, format="PNG")
        return {
            "timestamp": frame.timestamp,
            "age_seconds": max(0.0, captured_at - frame.timestamp),
            "device_timestamp": frame.device_timestamp,
            "timestamp_domain": frame.timestamp_domain,
            "frame_number": frame.frame_number,
            "frame_id": frame.frame_id,
            "serial_number": frame.serial_number,
            "encoding": frame.encoding,
            "intrinsics": asdict(frame.intrinsics),
            "depth_scale": frame.depth_scale,
            "png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    return {"rgb": encode(frames.rgb), "depth": encode(frames.depth)}


def decode_frames(payload: dict, received_after: float) -> CameraFrames:
    def decode(frame: dict) -> CameraImage:
        encoding = CameraEncoding(frame["encoding"])
        with Image.open(io.BytesIO(base64.b64decode(frame["png_base64"], validate=True))) as image:
            data = np.array(image, dtype=np.uint8 if encoding is CameraEncoding.RGB8 else np.uint16)
        calibration = dict(frame["intrinsics"])
        calibration["distortion_coefficients"] = tuple(calibration["distortion_coefficients"])
        return CameraImage(
            timestamp=received_after - frame["age_seconds"],
            device_timestamp=frame["device_timestamp"],
            timestamp_domain=frame["timestamp_domain"],
            frame_number=frame["frame_number"],
            frame_id=frame["frame_id"],
            serial_number=frame["serial_number"],
            encoding=encoding,
            data=data,
            intrinsics=CameraIntrinsics(**calibration),
            depth_scale=frame["depth_scale"],
        )

    return CameraFrames(rgb=decode(payload["rgb"]), depth=decode(payload["depth"]))
