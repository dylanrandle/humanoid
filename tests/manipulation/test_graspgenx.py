"""GraspGenX geometry, collision rejection, model boundary and pose conversion."""

# ruff: noqa: PLR2004 -- measured dimensions and explicit synthetic fixture sizes

from dataclasses import replace
from unittest.mock import MagicMock

import coal
import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.pick import GRASPGENX_CONFIG, PICK_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.manipulation.graspgenx_predictor import (
    TOOL_FROM_NATIVE,
    GraspGenXPredictor,
    convert_pose,
    estimate_width,
    gripper_geometry,
)
from humanoid.manipulation.pick_geometry import plan_pick
from humanoid.robots.base import Robot
from humanoid.types.pick import CameraPointCloud


@pytest.fixture(scope="module")
def robot():
    return Robot(TRISKEL_CONFIG)


@pytest.fixture
def predictor(robot):
    return GraspGenXPredictor(MagicMock(), robot, PICK_CONFIG, GRASPGENX_CONFIG)


def test_conditioning_matches_urdf_mimic_travel_and_is_base_invariant(robot):
    boxes, hulls = gripper_geometry(robot, GRASPGENX_CONFIG)
    half_angle = robot.get_gripper_limits()[0][1] / 2
    # Both fingers translate 11.5 mm/rad along their slightly tilted URDF axis.
    np.testing.assert_allclose(boxes[0] - boxes[6], half_angle * 0.023 * 0.999697, atol=1e-7)
    assert 0.048 < boxes[0] < 0.051
    assert 0.06 < boxes[2] < 0.08
    np.testing.assert_allclose(boxes[3:6], boxes[9:12], atol=1e-7)
    assert len(hulls) == 7


def test_native_pose_maps_to_tool_root_without_double_tip_offset(robot):
    native = pin.SE3(pin.rpy.rpyToMatrix(0.6, -0.3, 0.9), np.array([0.1, 0.2, 0.3]))
    base_camera = pin.SE3(pin.rpy.rpyToMatrix(-0.1, 0.2, -0.5), np.array([0.03, -0.04, 0.1]))
    cloud = CameraPointCloud(
        points=np.ones((100, 3), dtype=np.float32),
        rgb_pixels=np.zeros((100, 2), dtype=np.int64),
        base_from_camera=base_camera,
    )
    grasp = convert_pose(native.homogeneous, 0.8, PICK_CONFIG)
    config = replace(PICK_CONFIG, workspace_min=(-2.0, -2.0, -2.0), workspace_max=(2.0, 2.0, 2.0))
    plan = plan_pick(grasp, cloud, np.ones(100, dtype=bool), config, robot)
    expected = base_camera * native * pin.SE3(TOOL_FROM_NATIVE.T, np.zeros(3))
    np.testing.assert_allclose(plan.waypoints[1].pose.homogeneous, expected.homogeneous, atol=1e-12)
    assert [waypoint.name for waypoint in plan.waypoints] == ["approach", "grasp"]
    assert all(waypoint.gripper == 0 for waypoint in plan.waypoints)
    approach = plan.waypoints[1].pose.translation - plan.waypoints[0].pose.translation
    np.testing.assert_allclose(
        approach, config.approach_clearance * (base_camera * native).rotation[:, 2]
    )
    assert grasp.width is None


def test_observed_obstacle_on_approach_is_rejected(predictor):
    # A point inside the base hull, before the final target pose.
    hull, local = predictor.hulls[0]
    center = local * np.asarray(hull.center)
    center[2] -= PICK_CONFIG.approach_clearance / 2
    scene = coal.makeOctree(center.reshape(1, 3), GRASPGENX_CONFIG.collision_resolution)
    assert not predictor._clear_approach(pin.SE3.Identity(), scene)
    far_scene = coal.makeOctree(np.array([[10.0, 10.0, 10.0]]), 0.002)
    assert predictor._clear_approach(pin.SE3.Identity(), far_scene)


def test_only_segmented_points_reach_model_but_full_cloud_reaches_collisions(predictor):
    points = np.random.default_rng(4).uniform(1, 2, (200, 3)).astype(np.float32)
    points[:100] = object_points(predictor.conditioning)
    cloud = CameraPointCloud(
        points=points,
        rgb_pixels=np.full((200, 2), -1),
        base_from_camera=pin.SE3.Identity(),
    )
    region = np.arange(200) < 100
    predictor._infer = MagicMock(return_value=(np.eye(4)[None], np.array([0.8])))
    predictor._clear_approach = MagicMock(return_value=True)
    result = predictor.predict(cloud, region)
    np.testing.assert_array_equal(predictor._infer.call_args.args[0], points[region])
    assert len(result) == 1
    assert result[0].width == pytest.approx(0.025)
    assert isinstance(predictor._clear_approach.call_args.args[1], coal.OcTree)


def object_points(conditioning, width=0.025, offset=0.0):
    points = np.tile(conditioning[3:6], (100, 1))
    points[:, 0] += np.linspace(-width / 2, width / 2, len(points)) + offset
    return points


def test_width_is_invariant_to_camera_pose_and_uses_native_closing_axis(predictor):
    points = object_points(predictor.conditioning)
    pose = pin.SE3(pin.rpy.rpyToMatrix(0.4, -0.8, 1.2), np.array([0.1, -0.2, 0.3]))
    camera_points = points @ pose.rotation.T + pose.translation
    width = estimate_width(camera_points, pose, predictor.conditioning, PICK_CONFIG)
    assert width == pytest.approx(0.025)


def test_off_center_object_stops_at_nearer_finger_without_recentering(predictor, robot):
    points = object_points(predictor.conditioning, offset=0.003)
    native = pin.SE3.Identity()
    width = estimate_width(points, native, predictor.conditioning, PICK_CONFIG)
    assert width == pytest.approx(0.031)
    grasp = convert_pose(native.homogeneous, 0.9, PICK_CONFIG, width)
    cloud = CameraPointCloud(
        points=points,
        rgb_pixels=np.zeros((100, 2), dtype=np.int64),
        base_from_camera=pin.SE3.Identity(),
    )
    config = replace(PICK_CONFIG, workspace_min=(-2.0, -2.0, -2.0), workspace_max=(2.0, 2.0, 2.0))
    plan = plan_pick(grasp, cloud, np.ones(100, dtype=bool), config, robot)
    assert [waypoint.name for waypoint in plan.waypoints] == [
        "approach",
        "grasp",
        "squeeze",
        "lift",
    ]
    np.testing.assert_allclose(plan.waypoints[1].pose.translation, np.zeros(3), atol=1e-12)
    squeeze = plan.waypoints[2]
    target_gap = config.open_gap - squeeze.gripper * config.gap_per_radian
    assert target_gap == pytest.approx(0.027)
    assert 0 < squeeze.gripper < robot.get_gripper_limits()[0][1]
    assert plan.waypoints[3].gripper == squeeze.gripper


@pytest.mark.parametrize("bad", ["wide", "thin", "one_side", "sparse", "nan", "outlier"])
def test_unsupported_widths_have_no_full_close_fallback(predictor, bad):
    points = object_points(predictor.conditioning)
    if bad == "wide":
        points = object_points(predictor.conditioning, width=0.060)
    elif bad == "thin":
        points = object_points(predictor.conditioning, width=0.002)
    elif bad == "one_side":
        points[:, 0] += 0.020
    elif bad == "sparse":
        points = points[:2]
    elif bad == "nan":
        points[0, 0] = np.nan
    else:
        points[0, 0] = 0.100  # Never trim or clip a wide part into an accepted gap.
    assert estimate_width(points, pin.SE3.Identity(), predictor.conditioning, PICK_CONFIG) is None


def test_bad_width_is_rejected_before_collision_check(predictor):
    points = object_points(predictor.conditioning, width=0.060)
    cloud = CameraPointCloud(
        points=points,
        rgb_pixels=np.zeros((100, 2), dtype=np.int64),
        base_from_camera=pin.SE3.Identity(),
    )
    predictor._infer = MagicMock(return_value=(np.eye(4)[None], np.array([0.9])))
    predictor._clear_approach = MagicMock(return_value=True)
    assert predictor.predict(cloud, np.ones(100, dtype=bool)) == []
    predictor._clear_approach.assert_not_called()


@pytest.mark.parametrize("bad", ["score", "rotation", "bottom", "nan"])
def test_invalid_predictions_and_empty_target_do_not_become_commands(predictor, bad):
    matrix = np.eye(4)
    score = 0.9
    if bad == "score":
        score = 0.1
    elif bad == "rotation":
        matrix[0, 0] = -1
    elif bad == "bottom":
        matrix[3, 0] = 1
    else:
        matrix[0, 3] = np.nan
    cloud = CameraPointCloud(
        points=np.ones((100, 3), dtype=np.float32),
        rgb_pixels=np.zeros((100, 2), dtype=np.int64),
        base_from_camera=pin.SE3.Identity(),
    )
    predictor._infer = MagicMock(return_value=(matrix[None], np.array([score])))
    assert predictor.predict(cloud, np.ones(100, dtype=bool)) == []
    predictor._infer.reset_mock()
    with pytest.raises(ValueError, match="populated target"):
        predictor.predict(cloud, np.zeros(100, dtype=bool))
    predictor._infer.assert_not_called()


def test_empty_model_output_has_no_fallback(predictor):
    cloud = CameraPointCloud(
        points=np.ones((100, 3), dtype=np.float32),
        rgb_pixels=np.zeros((100, 2), dtype=np.int64),
        base_from_camera=pin.SE3.Identity(),
    )
    predictor._infer = MagicMock(return_value=(np.empty((0, 4, 4)), np.empty(0)))
    assert predictor.predict(cloud, np.ones(100, dtype=bool)) == []


def test_malformed_result_shape_fails_explicitly(predictor):
    cloud = CameraPointCloud(
        points=np.ones((100, 3), dtype=np.float32),
        rgb_pixels=np.zeros((100, 2), dtype=np.int64),
        base_from_camera=pin.SE3.Identity(),
    )
    predictor._infer = MagicMock(return_value=(np.eye(4), np.array(0.8)))
    with pytest.raises(ValueError, match="Unexpected GraspGenX output"):
        predictor.predict(cloud, np.ones(100, dtype=bool))


def test_generated_preview_includes_candidates_rejected_by_collision(predictor):
    cloud = CameraPointCloud(
        points=object_points(predictor.conditioning),
        rgb_pixels=np.full((100, 2), -1),
        base_from_camera=pin.SE3.Identity(),
    )
    predictor._infer = MagicMock(return_value=(np.eye(4)[None], np.array([0.9])))
    predictor._clear_approach = MagicMock(return_value=False)
    preview = MagicMock()
    result = predictor.predict(cloud, np.ones(100, dtype=bool), preview)
    assert result == []
    assert len(preview.call_args.args[0]) == 1
    assert preview.call_args.args[0][0].width == pytest.approx(0.025)
