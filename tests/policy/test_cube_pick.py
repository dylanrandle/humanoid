"""Native image registration and feedback-gated cube approach regression tests."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.camera import SIMULATED_CAMERA_CONFIG
from humanoid.config.cube_pick import CUBE_PICK_CONFIG, WOOD_CUBE_PICK_CONFIG, WOOD_CUBE_TEST_POSES
from humanoid.config.robot.triskel import HOME_POSITION, TRISKEL_CONFIG
from humanoid.policy.cube_pick import CubePickPolicy, detect_cube
from humanoid.robots.base import Robot
from humanoid.types.camera import CameraEncoding, CameraFrames, CameraImage, CameraIntrinsics
from humanoid.types.cube_pick import CubeDetection, CubePickPhase
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotState


@pytest.fixture(scope="module")
def robot():
    return Robot(TRISKEL_CONFIG)


def _frames(robot, q, center, yaw, config=CUBE_PICK_CONFIG):
    """Independent ray/box rendering: cameras have different FOVs and origins."""
    images = []
    rotation = pin.rpy.rpyToMatrix(0, 0, yaw)
    for frame, fov, encoding in (
        (SIMULATED_CAMERA_CONFIG.rgb_frame, 42, CameraEncoding.RGB8),
        (SIMULATED_CAMERA_CONFIG.depth_frame, 58, CameraEncoding.DEPTH16),
    ):
        height, width = 480, 640
        focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
        intr = CameraIntrinsics(
            width=width,
            height=height,
            fx=focal,
            fy=focal,
            cx=(width - 1) / 2,
            cy=(height - 1) / 2,
            distortion_model="none",
            distortion_coefficients=(),
        )
        v, u = np.indices((height, width))
        rays = np.stack(((u - intr.cx) / focal, (v - intr.cy) / focal, np.ones_like(u)), axis=-1)
        world_camera = robot.get_frame_pose(frame, q).copy()
        base = robot.get_base_pose(q)
        camera = world_camera if base is None else base.inverse() * world_camera
        origin = (camera.translation - center) @ rotation
        directions = rays @ camera.rotation.T @ rotation
        half = config.cube_size / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (-half - origin) / directions, (half - origin) / directions
        near = np.minimum(t0, t1).max(axis=-1)
        far = np.maximum(t0, t1).min(axis=-1)
        visible = (near > 0) & (near <= far)
        if encoding is CameraEncoding.RGB8:
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            pixels[visible] = [210, 20, 20]
        else:
            pixels = np.where(visible, np.rint(near * 1000), 0).astype(np.uint16)
        images.append(
            CameraImage(
                timestamp=0,
                device_timestamp=0,
                timestamp_domain="test",
                frame_number=1,
                frame_id=frame,
                serial_number="test",
                encoding=encoding,
                data=pixels,
                intrinsics=intr,
                depth_scale=0.001 if encoding is CameraEncoding.DEPTH16 else 0,
            )
        )
    return CameraFrames(rgb=images[0], depth=images[1])


@pytest.mark.parametrize("x,y,yaw", [(0, 0.28, 0), (-0.04, 0.27, 0.35), (0.025, 0.34, -0.7)])
@pytest.mark.parametrize("base_yaw", [0.0, 0.6])
def test_estimate_from_native_images_in_command_frame(robot, x, y, yaw, base_yaw):
    q = HOME_POSITION.copy()
    q[:4] = [0.4, -0.2, np.cos(base_yaw), np.sin(base_yaw)]
    center = np.array([x, y, CUBE_PICK_CONFIG.cube_size / 2])
    frames = _frames(robot, q, center, yaw)
    detection = detect_cube(frames, robot, q, CUBE_PICK_CONFIG)
    assert detection is not None
    np.testing.assert_allclose(detection.center, center, atol=0.002)
    assert abs(detection.yaw - yaw) < np.deg2rad(5)


def test_missing_depth_and_wrong_color_do_not_detect(robot):
    frames = _frames(robot, HOME_POSITION, np.array([0, 0.28, 0.0125]), 0)
    empty = replace(frames.depth, data=np.zeros_like(frames.depth.data))
    assert detect_cube(replace(frames, depth=empty), robot, HOME_POSITION, CUBE_PICK_CONFIG) is None
    blue = replace(frames.rgb, data=frames.rgb.data[..., ::-1].copy())
    assert detect_cube(replace(frames, rgb=blue), robot, HOME_POSITION, CUBE_PICK_CONFIG) is None


def _observation(timestamp=0.0, frames=None):
    return Observation(
        robot_state=RobotState(
            timestamp=timestamp,
            joint_positions=HOME_POSITION.copy(),
            joint_velocities=np.zeros(17),
            actuator_temperatures=np.zeros(11),
        ),
        camera_frames=frames,
    )


def test_stale_images_and_timeout_hold_open(robot):
    frames = _frames(robot, HOME_POSITION, np.array([0, 0.28, 0.0125]), 0)
    policy = CubePickPolicy(robot)
    action = policy(_observation(1.0, frames))
    assert policy.phase is CubePickPhase.SEARCH
    assert action.gripper_positions is not None
    assert action.gripper_positions[0] == CUBE_PICK_CONFIG.open_position
    policy(_observation(1.0 + CUBE_PICK_CONFIG.phase_timeout + 0.1, frames))
    assert policy.phase is CubePickPhase.FAILED
    assert policy.failure_reason is not None


def test_waits_for_measured_pose_before_squeezing_and_resets(robot):
    policy = CubePickPolicy(robot)
    frames = _frames(robot, HOME_POSITION, np.array([0, 0.28, 0.0125]), 0)
    detection = CubeDetection(center=np.array([0, 0.28, 0.0125]), yaw=0, point_count=100)
    with patch("humanoid.policy.cube_pick.detect_cube", return_value=detection):
        approach = policy(_observation(frames=frames))
    assert policy.phase is CubePickPhase.APPROACH
    waiting = policy(_observation(0.5))
    assert policy.phase is CubePickPhase.APPROACH
    assert waiting.gripper_positions is not None
    assert waiting.gripper_positions[0] == CUBE_PICK_CONFIG.open_position
    with patch.object(robot, "get_tool_command_pose", return_value=approach.tool_pose):
        descend = policy(_observation(0.6))
    assert policy.phase is CubePickPhase.DESCEND
    assert descend.tool_pose.translation[2] < approach.tool_pose.translation[2]
    assert descend.gripper_positions[0] == CUBE_PICK_CONFIG.open_position
    with patch.object(robot, "get_tool_command_pose", return_value=descend.tool_pose):
        squeeze = policy(_observation(0.8))
        assert policy.phase is CubePickPhase.SQUEEZE
        assert squeeze.gripper_positions[0] == CUBE_PICK_CONFIG.closed_position
        policy(_observation(0.8 + CUBE_PICK_CONFIG.squeeze_seconds + 0.01))
    assert policy.is_done
    inactive = replace(_observation(3), mode=Mode.IDLE)
    assert policy(inactive).tool_pose is None
    assert policy.phase is CubePickPhase.SEARCH
    assert policy.detection is None


@pytest.mark.parametrize("x,y,yaw", WOOD_CUBE_TEST_POSES)
def test_wood_cube_geometry_does_not_require_red_or_top_face(robot, x, y, yaw):
    config = WOOD_CUBE_PICK_CONFIG
    center = np.array([x, y, config.ground_height + config.cube_size / 2])
    frames = _frames(robot, HOME_POSITION, center, yaw, config)
    # All RGB samples are gray; the detector must use depth geometry.
    frames = replace(frames, rgb=replace(frames.rgb, data=np.full_like(frames.rgb.data, 120)))
    detection = detect_cube(frames, robot, HOME_POSITION, config)
    assert detection is not None
    np.testing.assert_allclose(detection.center, center, atol=0.003)
    assert abs(detection.yaw - yaw) < np.deg2rad(5)


def test_wood_cube_outside_workspace_is_rejected(robot):
    config = WOOD_CUBE_PICK_CONFIG
    frames = _frames(robot, HOME_POSITION, np.array([0, 0.42, -0.0095]), 0.3, config)
    assert detect_cube(frames, robot, HOME_POSITION, config) is None
