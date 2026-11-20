"""Capture one live robot observation without a policy or command publisher."""

import time

from humanoid.config.camera import HARDWARE_SNAPSHOT_MAX_AGE_SECONDS
from humanoid.config.remote_control import get_remote_control_policy_config
from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.remote_control.observation import measured_geometry, optical_to_base
from humanoid.remote_control.serialization import encode_frames, encode_state
from humanoid.robots.base import Robot
from humanoid.types.camera import CameraEncoding, CameraFrames
from humanoid.types.orchestrator import OrchestratorError


def hardware_snapshot(robot: Robot) -> dict:
    subscriber = Subscriber([Topic.ROBOT_STATE, Topic.CAMERA_RGB, Topic.CAMERA_DEPTH])
    try:
        deadline = time.perf_counter() + 2
        state_timeout = get_remote_control_policy_config().feedback_timeout
        while time.perf_counter() < deadline:
            rgb = subscriber.receive(Topic.CAMERA_RGB, timeout=100)
            depth = subscriber.receive(Topic.CAMERA_DEPTH, timeout=100)
            state = subscriber.receive(Topic.ROBOT_STATE, timeout=100)
            now = time.perf_counter()
            if (
                rgb is not None
                and depth is not None
                and state is not None
                and rgb.encoding is CameraEncoding.RGB8
                and depth.encoding is CameraEncoding.DEPTH16
                and rgb.timestamp == depth.timestamp
                and rgb.serial_number == depth.serial_number
                and rgb.serial_number not in {"", "simulation"}
                and 0 <= now - rgb.timestamp <= HARDWARE_SNAPSHOT_MAX_AGE_SECONDS
                and 0 <= now - state.timestamp <= state_timeout
            ):
                frames = encode_frames(CameraFrames(rgb=rgb, depth=depth))
                for stream in frames.values():
                    stream["optical_to_base"] = optical_to_base(robot, state, stream["frame_id"])
                return {
                    "runtime": "real",
                    "robot": robot.config.name,
                    "state": {**encode_state(state), **measured_geometry(robot, state)},
                    "frames": frames,
                }
        raise OrchestratorError("No fresh paired hardware RGB-D frames and robot state arrived.")
    finally:
        subscriber.close()
