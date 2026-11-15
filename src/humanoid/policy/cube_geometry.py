"""Fit a known-size cube from connected depth samples on its vertical faces."""

import numpy as np

from humanoid.types.cube_pick import CubeDetection, CubePickConfig


def detect_depth_cube(
    points: np.ndarray,
    camera_position: np.ndarray,
    config: CubePickConfig,
) -> CubeDetection | None:
    """Require one cube-shaped component; reject ambiguous/cluttered candidates.

    Ground and top-edge samples are excluded. A known square's two visible
    vertical faces define its center, even when its top is not visible. This is
    geometric matching, not recognition of arbitrary objects as wooden cubes.
    """
    margin = max(config.top_tolerance * 2, config.cube_size * 0.15)
    lower = config.ground_height + margin
    upper = config.ground_height + config.cube_size - margin
    # Include a half-size border so the full surface of an in-bounds cube can fit.
    xy_min = np.asarray(config.workspace_min[:2]) - config.cube_size / 2
    xy_max = np.asarray(config.workspace_max[:2]) + config.cube_size / 2
    inside = (points[:, 2] > lower) & (points[:, 2] < upper)
    inside &= np.all((points[:, :2] > xy_min) & (points[:, :2] < xy_max), axis=1)
    candidates = []
    for component in _components(points[inside], cell_size=config.cube_size / 8):
        if len(component) < config.minimum_points:
            continue
        if np.ptp(component[:, 2]) < (upper - lower) * 0.6:
            continue
        detection = _fit_faces(component, camera_position, config)
        if detection is not None:
            candidates.append(detection)
    return candidates[0] if len(candidates) == 1 else None


def _components(points: np.ndarray, cell_size: float) -> list[np.ndarray]:
    """Connected occupied XY cells; avoid depending on a vision/clustering package."""
    cells: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(np.floor(points[:, :2] / cell_size).astype(int)):
        cells.setdefault((int(cell[0]), int(cell[1])), []).append(index)
    components = []
    while cells:
        seed, indices = cells.popitem()
        pending = [seed]
        while pending:
            x, y = pending.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (x + dx, y + dy)
                    if neighbor in cells:
                        indices.extend(cells.pop(neighbor))
                        pending.append(neighbor)
        components.append(points[indices])
    return components


def _fit_faces(
    points: np.ndarray,
    camera_position: np.ndarray,
    config: CubePickConfig,
) -> CubeDetection | None:
    best = None
    for yaw in np.deg2rad(np.arange(-45, 45)):
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        local = points[:, :2] @ rotation
        low, high = np.quantile(local, [0.01, 0.99], axis=0)
        extent = high - low
        if np.max(extent) < config.cube_size * 0.75 or np.max(extent) > config.cube_size * 1.15:
            continue
        camera = camera_position[:2] @ rotation
        facing_low = camera < (low + high) / 2
        edges = np.where(facing_low, low, high)
        center = edges + np.where(facing_low, 1, -1) * config.cube_size / 2
        residual = np.min(np.abs(local - edges), axis=1)
        score = float(np.mean(residual**2))
        if best is None or score < best[0]:
            best = (score, rotation @ center, float(yaw), float(np.quantile(residual, 0.95)))
    if best is None or best[3] > config.top_tolerance:
        return None
    _, xy, yaw, _ = best
    center = np.array([*xy, config.ground_height + config.cube_size / 2])
    if np.any(center < config.workspace_min) or np.any(center > config.workspace_max):
        return None
    return CubeDetection(center=center, yaw=yaw, point_count=len(points))
