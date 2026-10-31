"""Euclidean shortest routes to target-facing approach poses in planar free space.

A visibility graph gives exact shortest paths for the conservative polygon map.
Approach positions lie on a continuous standoff circle, not a sampled heading grid.
"""

import heapq

import numpy as np
from scipy.spatial import ConvexHull

from humanoid.types.tasks.cleanup import NavigationMap, NavigationObstacle, NavigationPlan

GEOMETRY_TOLERANCE = 1e-9
MIN_OBSTACLE_SIDES = 8


def inflated_obstacle(points: np.ndarray, radius: float, sides: int) -> NavigationObstacle:
    """Minkowski sum with a circumscribed regular polygon enclosing the robot disk.

    The polygon never underestimates the circular swept footprint. At 32 sides
    and radius 0.215 m its extra radial conservatism is at most 1.04 mm.
    """
    if radius <= 0 or sides < MIN_OBSTACLE_SIDES:
        raise ValueError("Obstacle inflation needs a positive radius and at least eight sides.")
    angles = (np.arange(sides) + 0.5) * 2 * np.pi / sides
    offsets = np.c_[np.cos(angles), np.sin(angles)] * radius / np.cos(np.pi / sides)
    cloud = (points[:, None, :2] + offsets).reshape(-1, 2)
    hull = ConvexHull(cloud)
    return NavigationObstacle(cloud[hull.vertices], hull.equations.copy())


def free_point(world: NavigationMap, point: np.ndarray) -> bool:
    if np.any(point < world.lower - GEOMETRY_TOLERANCE) or np.any(
        point > world.upper + GEOMETRY_TOLERANCE
    ):
        return False
    return all(
        np.any(
            obstacle.halfspaces[:, :2] @ point + obstacle.halfspaces[:, 2] >= -GEOMETRY_TOLERANCE
        )
        for obstacle in world.obstacles
    )


def clear_segment(world: NavigationMap, start: np.ndarray, end: np.ndarray) -> bool:
    """Exact segment/interior clipping; boundary tangency is allowed at the safety margin."""
    if not free_point(world, start) or not free_point(world, end):
        return False
    for obstacle in world.obstacles:
        value = obstacle.halfspaces[:, :2] @ start + obstacle.halfspaces[:, 2]
        delta = obstacle.halfspaces[:, :2] @ (end - start)
        parallel = np.abs(delta) < GEOMETRY_TOLERANCE
        if np.any(parallel & (value >= -GEOMETRY_TOLERANCE)):
            continue
        crossing = -value[~parallel] / delta[~parallel]
        entering = crossing[delta[~parallel] < 0]
        leaving = crossing[delta[~parallel] > 0]
        lower = max(0.0, float(entering.max(initial=-np.inf)))
        upper = min(1.0, float(leaving.min(initial=np.inf)))
        if lower < upper - GEOMETRY_TOLERANCE:
            return False
    return True


def _circle_intersections(
    vertices: np.ndarray, center: np.ndarray, radius: float
) -> list[np.ndarray]:
    intersections = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        delta, offset = end - start, start - center
        a, b, c = delta @ delta, 2 * (offset @ delta), offset @ offset - radius**2
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            continue
        fractions = (
            (-b - np.sqrt(discriminant)) / (2 * a),
            (-b + np.sqrt(discriminant)) / (2 * a),
        )
        intersections.extend(
            start + np.clip(fraction, 0, 1) * delta
            for fraction in fractions
            if -GEOMETRY_TOLERANCE <= fraction <= 1 + GEOMETRY_TOLERANCE
        )
    return intersections


def _radial_approach(origin: np.ndarray, target: np.ndarray, radius: float) -> np.ndarray:
    offset = origin - target
    distance = np.linalg.norm(offset)
    if distance < GEOMETRY_TOLERANCE:
        raise ValueError("Base coincides with the pickup/place target.")
    return target + radius * offset / distance


def _approach_endpoints(
    world: NavigationMap,
    target: np.ndarray,
    pickup_offset: np.ndarray,
    headings: np.ndarray | None,
) -> list[np.ndarray]:
    if headings is not None:
        return [
            target
            - np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ pickup_offset
            for yaw in headings
        ]
    radius = float(np.linalg.norm(pickup_offset))
    border = np.array(
        [
            world.lower,
            [world.upper[0], world.lower[1]],
            world.upper,
            [world.lower[0], world.upper[1]],
        ]
    )
    endpoints = _circle_intersections(border, target, radius)
    for obstacle in world.obstacles:
        endpoints.extend(_circle_intersections(obstacle.vertices, target, radius))
    return endpoints


def shortest_approach(
    world: NavigationMap,
    start: np.ndarray,
    target: np.ndarray,
    pickup_offset: np.ndarray,
    *,
    approach_yaws: np.ndarray | None = None,
) -> NavigationPlan:
    """Choose the shortest collision-free route AND approach side.

    A shortest path bends only at visible obstacle vertices. Its last segment
    meets the approach circle radially, or at an endpoint of a feasible arc
    (a circle/polygon intersection). Testing these endpoints is exhaustive for
    this polygon map, without restricting the robot to fixed approach headings.
    When approach_yaws is supplied, optimize over those grasp-compatible poses
    instead; each candidate still uses an exact polygon visibility-graph route.
    """
    radius = float(np.linalg.norm(pickup_offset))
    if radius <= 0 or not free_point(world, start):
        raise ValueError("Invalid standoff or base starting inside an inflated obstacle.")
    direct = _radial_approach(start, target, radius)
    if approach_yaws is None and clear_segment(world, start, direct):
        return _plan((direct,), target, pickup_offset, float(np.linalg.norm(direct - start)))

    vertices = [start.copy()]
    for obstacle in world.obstacles:
        vertices.extend(point for point in obstacle.vertices if free_point(world, point))
    nodes = np.asarray(vertices)
    endpoints = _approach_endpoints(world, target, pickup_offset, approach_yaws)
    endpoints = [point for point in endpoints if free_point(world, point)]
    distances = np.full(len(nodes), np.inf)
    distances[0] = 0.0
    previous: dict[int, int] = {}
    queue = [(0.0, 0)]
    best, final_node, final_point = np.inf, 0, direct
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != distances[current] or cost >= best:
            continue
        point = nodes[current]
        candidates = (
            [_radial_approach(point, target, radius), *endpoints]
            if approach_yaws is None
            else endpoints
        )
        for end in candidates:
            length = cost + float(np.linalg.norm(end - point))
            if length < best and clear_segment(world, point, end):
                best, final_node, final_point = length, current, end
        for other, neighbor in enumerate(nodes):
            length = cost + float(np.linalg.norm(neighbor - point))
            lower_bound = max(0.0, float(np.linalg.norm(neighbor - target)) - radius)
            if (
                length < distances[other]
                and length + lower_bound < best
                and clear_segment(world, point, neighbor)
            ):
                distances[other] = length
                previous[other] = current
                heapq.heappush(queue, (length, other))
    if not np.isfinite(best):
        raise ValueError("No collision-free route to a target-facing approach pose.")
    route = [final_point]
    while final_node:
        route.append(nodes[final_node])
        final_node = previous[final_node]
    return _plan(tuple(reversed(route)), target, pickup_offset, float(best))


def _plan(
    waypoints: tuple[np.ndarray, ...], target: np.ndarray, pickup_offset: np.ndarray, length: float
) -> NavigationPlan:
    toward = target - waypoints[-1]
    yaw = np.arctan2(toward[1], toward[0]) - np.arctan2(pickup_offset[1], pickup_offset[0])
    return NavigationPlan(waypoints, target.copy(), float(wrap_angle(yaw)), length)


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))
