"""Geometric route optimality, clearance, and target-facing approach regressions."""

from dataclasses import replace
from itertools import pairwise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from humanoid.simulation.tasks.cleanup.motion import CleanupMotion
from humanoid.simulation.tasks.cleanup.navigation import (
    clear_segment,
    free_point,
    inflated_obstacle,
    shortest_approach,
    wrap_angle,
)
from humanoid.simulation.tasks.cleanup.task import CleanupTask
from humanoid.types.robot import RobotJointCommand
from humanoid.types.tasks.cleanup import NavigationMap, NavigationObstacle

GEOMETRY_TOLERANCE = 1e-9


def rectangle(lower, upper):
    vertices = np.array([lower, [upper[0], lower[1]], upper, [lower[0], upper[1]]])
    return NavigationObstacle(vertices, ConvexHull(vertices).equations)


@pytest.mark.parametrize("start", [[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [1.0, 1.0]])
def test_unobstructed_approach_attains_euclidean_lower_bound_and_faces_target(start):
    world = NavigationMap(np.full(2, -3.0), np.full(2, 3.0), ())
    start, target, offset = np.array(start), np.zeros(2), np.array([0.0, 0.305])
    plan = shortest_approach(world, start, target, offset)
    assert len(plan.waypoints) == 1
    assert plan.length == pytest.approx(np.linalg.norm(start) - np.linalg.norm(offset))
    cosine, sine = np.cos(plan.yaw), np.sin(plan.yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    assert plan.waypoints[-1] + rotation @ offset == pytest.approx(target)


def test_blocked_route_matches_analytic_shortest_path_around_rectangle():
    world = NavigationMap(np.full(2, -3.0), np.full(2, 3.0), (rectangle([-0.5, -0.5], [0.5, 0.5]),))
    start, target = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    plan = shortest_approach(world, start, target, np.array([0.0, 0.25]))
    # Two diagonal corner segments, one unit of rectangle edge, minus standoff.
    assert plan.length == pytest.approx(2 * np.hypot(1.5, 0.5) + 1 - 0.25)
    assert len(plan.waypoints) == 3  # noqa: PLR2004
    route = (start, *plan.waypoints)
    assert all(clear_segment(world, a, b) for a, b in pairwise(route))
    assert not clear_segment(world, start, target)


def test_face_grasp_constraint_selects_shortest_of_feasible_approaches():
    world = NavigationMap(np.full(2, -3.0), np.full(2, 3.0), ())
    start, target = np.array([-1.5, -0.7]), np.zeros(2)
    offset, headings = np.array([0.0, 0.305]), np.arange(4) * np.pi / 2 + 0.3
    plan = shortest_approach(world, start, target, offset, approach_yaws=headings)
    candidates = np.c_[np.sin(headings), -np.cos(headings)] * np.linalg.norm(offset)
    assert plan.length == pytest.approx(np.min(np.linalg.norm(candidates - start, axis=1)))
    assert min(abs(wrap_angle(plan.yaw - yaw)) for yaw in headings) < GEOMETRY_TOLERANCE


def test_inflated_geometry_encloses_the_entire_swept_disk():
    points = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
    obstacle = inflated_obstacle(points, 0.215, 32)
    angles = np.linspace(0, 2 * np.pi, 1000)
    swept = (points[:, None, :] + 0.215 * np.c_[np.cos(angles), np.sin(angles)]).reshape(-1, 2)
    assert (
        np.max(swept @ obstacle.halfspaces[:, :2].T + obstacle.halfspaces[:, 2])
        < GEOMETRY_TOLERANCE
    )


def test_unreachable_approach_and_occupied_start_are_rejected():
    world = NavigationMap(np.full(2, -1.0), np.full(2, 1.0), (rectangle([-0.1, -0.1], [0.1, 0.1]),))
    with pytest.raises(ValueError, match="starting inside"):
        shortest_approach(world, np.zeros(2), np.array([0.8, 0.0]), np.array([0.0, 0.305]))
    with pytest.raises(ValueError, match="No collision-free route"):
        shortest_approach(
            world, np.array([-0.5, 0.0]), np.array([3.0, 0.0]), np.array([0.0, 0.305])
        )


def test_acquire_selects_nearest_target_without_waiting_for_visibility():
    task = CleanupTask()
    engine = task.create_engine(0)
    runtime = MagicMock(engine=engine)
    runtime.state.return_value = engine.read_robot_state(0)
    motion = CleanupMotion(
        runtime, task.arm_config, task.mobile_config, task.scene, task.view_config
    )
    with patch.object(
        motion.view, "best_view", side_effect=AssertionError("Image visibility is diagnostic")
    ):
        assert motion.acquire(["green_block", "orange_car", "red_block"]) == "orange_car"
    assert motion.view_target == "orange_car"
    assert motion.acquire(["cleanup_bin"]) == "cleanup_bin"
    runtime.step.assert_not_called()


def test_cleanup_base_tracks_unfiltered_yaw_steps_without_oscillation():
    engine = CleanupTask().create_engine(0)
    root = engine.binding.root
    root_v = engine.robot.get_root_v_slice()
    assert root is not None and root_v is not None
    positions = engine.read_robot_state(0).joint_positions
    velocities = np.zeros(engine.robot.model.nv)
    for target in (0.4, -0.4, 0.0):
        velocities[root_v] = [0, 0, target]
        engine.apply_joint_command(RobotJointCommand(0, positions, velocities))
        measured = []
        for _ in range(500):
            engine.step()
            measured.append(engine.data.qvel[root.qvel_addresses[2]])
        # Saturating the old scene-specific gain produced alternating +/-7 rad/s.
        assert np.max(np.abs(measured)) < 0.5  # noqa: PLR2004
        assert measured[-1] == pytest.approx(target, abs=0.005)
    assert not any(warning.number for warning in engine.data.warning)


def test_rotating_base_physically_follows_a_detour_without_cutting_obstacle_corners(stack_runtime):
    task = CleanupTask()
    runtime = stack_runtime(task)
    engine = runtime.engine
    motion = CleanupMotion(
        runtime, task.arm_config, task.mobile_config, task.scene, task.view_config
    )
    motion.acquire(["green_block"])
    world = task.navigation_map(engine)
    target = task.objects(engine)[2].pose.translation[:2]
    plan = shortest_approach(
        world, motion.base_pose()[:2], target, np.asarray(task.mobile_config.pickup_offset)
    )
    assert len(plan.waypoints) > 1  # The car obstructs the direct route to green.
    # Allow the specified 3 mm tracking tolerance while retaining 10 mm margin.
    task.mobile_config = replace(task.mobile_config, clearance=0.010)
    swept_world = task.navigation_map(engine)
    points = []

    def observe(state, command, index):
        if index % 5 == 0:
            points.append(motion.base_pose()[:2])

    runtime.observe = observe
    motion.drive(plan)
    assert all(free_point(swept_world, point) for point in points)
    assert abs(wrap_angle(motion.base_pose()[2] - plan.yaw)) < task.mobile_config.yaw_tolerance
    assert (
        np.linalg.norm(motion.base_pose()[:2] - plan.waypoints[-1])
        < task.mobile_config.position_tolerance
    )
