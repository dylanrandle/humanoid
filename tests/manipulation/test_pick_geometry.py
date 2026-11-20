"""Native camera registration and Canonical-grasp-to-Triskel pose conversion."""

from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.pick import PICK_CONFIG
from humanoid.manipulation.pick_geometry import (
    distort,
    object_region,
    plan_pick,
    reconstruct_scene,
)
from humanoid.manipulation.pick_motion import advance_pose
from humanoid.types.camera import CameraEncoding, CameraFrames, CameraImage, CameraIntrinsics
from humanoid.types.pick import CameraPointCloud, GraspProposal, ObjectMask


def frames(depth_pixels=None):
    intrinsics = CameraIntrinsics(
        width=5,
        height=3,
        fx=10,
        fy=10,
        cx=2,
        cy=1,
        distortion_model="none",
        distortion_coefficients=(0.0,) * 5,
    )
    rgb = CameraImage(
        timestamp=10.0,
        device_timestamp=9.0,
        timestamp_domain="device",
        frame_number=1,
        serial_number="test",
        intrinsics=intrinsics,
        frame_id="rgb",
        encoding=CameraEncoding.RGB8,
        data=np.zeros((3, 5, 3), dtype=np.uint8),
    )
    depth = np.zeros((3, 5), dtype=np.uint16)
    depth[1, 2] = 500
    return CameraFrames(
        rgb=rgb,
        depth=replace(
            rgb,
            frame_id="depth",
            encoding=CameraEncoding.DEPTH16,
            data=depth if depth_pixels is None else depth_pixels,
            depth_scale=0.001,
        ),
    )


def camera_robot():
    robot = MagicMock()
    base = pin.SE3(pin.rpy.rpyToMatrix(0, 0, 0.5), np.array([4.0, 3.0, 0.0]))
    robot.get_base_pose.return_value = base
    poses = {"depth": base, "rgb": base * pin.SE3(np.eye(3), np.array([0.05, 0, 0]))}
    robot.get_frame_pose.side_effect = lambda frame, q: poses[frame]
    return robot


def test_native_depth_uses_optical_z_and_rgb_baseline_and_base_frame():
    cloud = reconstruct_scene(frames(), camera_robot(), np.zeros(1))
    points, pixels = cloud.base_points, cloud.rgb_pixels
    np.testing.assert_allclose(points, [[0, 0, 0.5]], atol=1e-12)
    # 5 cm baseline at 0.5 m moves the central depth pixel one RGB column left.
    np.testing.assert_array_equal(pixels, [[1, 1]])


def test_invalid_depth_is_not_used():
    cloud = reconstruct_scene(
        frames(np.zeros((3, 5), dtype=np.uint16)), camera_robot(), np.zeros(1)
    )
    points, pixels = cloud.base_points, cloud.rgb_pixels
    assert points.shape == (0, 3)
    assert pixels.shape == (0, 2)


def test_camera_pair_mismatch_is_rejected():
    pair = frames()
    with pytest.raises(ValueError, match="same camera frameset"):
        reconstruct_scene(
            replace(pair, depth=replace(pair.depth, timestamp=11)), camera_robot(), np.zeros(1)
        )


def test_brown_distortion_roundtrip_and_unknown_calibration():
    intr = replace(
        frames().depth.intrinsics,
        distortion_model="brown_conrady",
        distortion_coefficients=(0.12, -0.03, 0.002, -0.001, 0.01),
    )
    points = np.array([[0.1, -0.3], [-0.5, 0.4], [0, 0]])
    np.testing.assert_allclose(
        distort(distort(points, intr, inverse=False), intr, inverse=True), points, atol=1e-6
    )
    with pytest.raises(ValueError, match="Unsupported"):
        distort(points, replace(intr, distortion_model="fish_eye"), inverse=True)


def test_nearest_rgb_depth_occludes_background():
    pair = frames()
    # Two depth rays quantize to one low-resolution RGB pixel.
    intr = replace(pair.depth.intrinsics, fx=100)
    data = np.zeros((3, 5), dtype=np.uint16)
    data[1, 2:4] = [200, 800]
    pair = replace(pair, depth=replace(pair.depth, intrinsics=intr, data=data))
    robot = camera_robot()
    robot.get_frame_pose.side_effect = lambda frame, q: pin.SE3.Identity()
    robot.get_base_pose.return_value = None
    cloud = reconstruct_scene(pair, robot, np.zeros(1))
    assert cloud.points.shape == (2, 3)  # Hidden surfaces still matter for collisions.
    np.testing.assert_array_equal(cloud.rgb_pixels, [[1, 2], [-1, -1]])


def grasp_fixture():
    base_from_camera = pin.SE3(pin.rpy.rpyToMatrix(0.3, -0.2, 0.6), np.array([0.03, 0.1, 0.08]))
    base_points = np.random.default_rng(0).uniform(
        [-0.025, 0.30, -0.026], [0.025, 0.32, -0.005], (60, 3)
    )
    points = (base_points - base_from_camera.translation) @ base_from_camera.rotation
    cloud = CameraPointCloud(
        points=points.astype(np.float32),
        rgb_pixels=np.full((60, 2), 2),
        base_from_camera=base_from_camera,
    )
    base_grasp = pin.SE3(
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]), np.array([0.0, 0.31, 0.0])
    )
    grasp = GraspProposal(
        pose=base_from_camera.inverse() * base_grasp,
        score=0.9,
        width=0.025,
        height=0.04,
        depth=0.02,
    )
    robot = MagicMock()
    robot.get_gripper_limits.return_value = [(0, 2.2028)]
    return grasp, cloud, np.ones(60, dtype=bool), robot


def test_pick_preserves_predicted_axes_depth_preload_and_lift():
    grasp, cloud, region, robot = grasp_fixture()
    plan = plan_pick(grasp, cloud, region, PICK_CONFIG, robot)
    approach, descend, squeeze, lift = plan.waypoints
    np.testing.assert_allclose(descend.pose.rotation[:, 1], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(descend.pose.rotation[:, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(
        descend.pose * np.array(PICK_CONFIG.tool_tip_offset), [0, 0.31, -0.02], atol=1e-12
    )
    deeper = plan_pick(replace(grasp, depth=0.025), cloud, region, PICK_CONFIG, robot)
    assert deeper.waypoints[1].pose.translation[2] == pytest.approx(
        descend.pose.translation[2] - 0.005
    )
    assert squeeze.gripper > (PICK_CONFIG.open_gap - grasp.width) / PICK_CONFIG.gap_per_radian
    assert lift.gripper == squeeze.gripper
    assert lift.pose.translation[2] - descend.pose.translation[2] == pytest.approx(0.08)
    assert approach.gripper == descend.gripper == 0


def test_approach_retracts_along_prediction_instead_of_world_vertical():
    grasp, cloud, region, robot = grasp_fixture()
    base_grasp = cloud.base_from_camera * grasp.pose
    tilt = pin.rpy.rpyToMatrix(0.1, 0, 0)
    base_grasp.rotation = tilt @ base_grasp.rotation
    grasp = replace(grasp, pose=cloud.base_from_camera.inverse() * base_grasp)
    plan = plan_pick(grasp, cloud, region, PICK_CONFIG, robot)
    np.testing.assert_allclose(
        plan.waypoints[0].pose.translation - plan.waypoints[1].pose.translation,
        -base_grasp.rotation[:, 0] * PICK_CONFIG.approach_clearance,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        plan.waypoints[1].pose.rotation[:, 1], base_grasp.rotation[:, 0], atol=1e-12
    )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"width": 0.06}, "width"),
        ({"width": 0.002}, "width"),
    ],
)
def test_impossible_grasps_are_rejected(change, message):
    grasp, cloud, region, robot = grasp_fixture()
    with pytest.raises(ValueError, match=message):
        plan_pick(replace(grasp, **change), cloud, region, PICK_CONFIG, robot)


def test_arbitrary_grasp_orientation_is_preserved():
    grasp, cloud, region, robot = grasp_fixture()
    # Horizontal approach, without a manually imposed downward cone.
    base_grasp = pin.SE3(np.eye(3), np.array([0.1, 0.30, 0.08]))
    grasp = replace(grasp, pose=cloud.base_from_camera.inverse() * base_grasp)
    plan = plan_pick(grasp, cloud, region, PICK_CONFIG, robot)
    np.testing.assert_allclose(plan.waypoints[1].pose.rotation[:, 1], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(
        plan.waypoints[1].pose * np.array(PICK_CONFIG.tool_tip_offset),
        base_grasp.translation + grasp.depth * base_grasp.rotation[:, 0],
        atol=1e-12,
    )


def test_workspace_limits_remain_execution_guard():
    grasp, cloud, region, robot = grasp_fixture()
    pose = grasp.pose.copy()
    pose.translation += np.array([1.0, 0, 0])
    with pytest.raises(ValueError, match="workspace"):
        plan_pick(replace(grasp, pose=pose), cloud, region, PICK_CONFIG, robot)


def test_object_region_preserves_model_mask_without_shape_or_floor_filters():
    _, cloud, _, _ = grasp_fixture()
    mask_pixels = np.zeros((5, 5), dtype=bool)
    mask_pixels[0, 0] = True  # A boundary pixel must not disappear through erosion.
    mask = ObjectMask(mask=mask_pixels, score=0.9, box=(0, 0, 5, 5))
    cloud = replace(
        cloud,
        rgb_pixels=np.zeros_like(cloud.rgb_pixels),
        base_from_camera=pin.SE3(np.eye(3), np.array([10.0, 0.0, -10.0])),
    )
    assert object_region(mask, cloud, PICK_CONFIG).all()


def test_empty_region_cannot_request_grasp_inference():
    _, cloud, _, _ = grasp_fixture()
    empty = ObjectMask(mask=np.zeros((5, 5), dtype=bool), score=0.9, box=(0, 0, 5, 5))
    with pytest.raises(ValueError, match="Too few"):
        object_region(empty, cloud, PICK_CONFIG)
    mask = replace(empty, mask=np.ones((5, 5), dtype=bool))
    assert object_region(mask, cloud, PICK_CONFIG).all()
    pixels = cloud.rgb_pixels.copy()
    pixels[0] = [-1, -1]
    region = object_region(mask, replace(cloud, rgb_pixels=pixels), PICK_CONFIG)
    assert not region[0]  # No accidental mask[-1, -1] indexing.
    assert region[1:].all()


def test_motion_limits_translation_rotation_and_no_overshoot():
    start = pin.SE3.Identity()
    target = pin.SE3(pin.rpy.rpyToMatrix(0, 0, 1), np.array([1.0, 0, 0]))
    step = advance_pose(start, target, 0.1, PICK_CONFIG)
    assert np.linalg.norm(step.translation) == pytest.approx(PICK_CONFIG.linear_speed * 0.1)
    assert np.linalg.norm(pin.log3(step.rotation)) == pytest.approx(PICK_CONFIG.angular_speed * 0.1)
    np.testing.assert_allclose(
        advance_pose(start, step, 1, PICK_CONFIG).homogeneous, step.homogeneous
    )
