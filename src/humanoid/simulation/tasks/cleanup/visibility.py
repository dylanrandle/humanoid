"""Optional four-camera visibility diagnostics."""

from dataclasses import replace
from itertools import product

import mujoco
import numpy as np

from humanoid.config.tasks.cleanup import WRIST_CAMERA_NAME
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.objects import body_bounding_box, read_object
from humanoid.types.tasks.cleanup import CleanupSceneConfig, CleanupViewConfig, TargetVisibility


def _clip_box_at_near_plane(points: np.ndarray, near: float) -> np.ndarray:
    """Clip the 12 OBB edges before perspective division, as the renderer does."""
    vertices = list(points[points[:, 2] >= near])
    for index in range(8):
        for bit in (1, 2, 4):
            other = index ^ bit
            if other <= index:
                continue
            first, second = points[index], points[other]
            if (first[2] < near) != (second[2] < near):
                fraction = (near - first[2]) / (second[2] - first[2])
                vertices.append(first + fraction * (second - first))
    return np.asarray(vertices).reshape(-1, 3)


class CleanupView:
    def __init__(
        self,
        engine: NativeMujocoEngine,
        scene: CleanupSceneConfig,
        config: CleanupViewConfig,
    ):
        self.engine, self.config = engine, config
        self.cameras = scene.cameras
        self.bin_corners = np.array(
            [[*scene.bin_position, scene.floor_height + scene.bin_height / 2]]
        ) + np.asarray(list(product((-1, 1), repeat=3))) * [
            *scene.bin_half_size,
            scene.bin_height / 2,
        ]
        self.bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def measure(
        self, name: str, *, held: bool = False, camera_name: str = WRIST_CAMERA_NAME
    ) -> TargetVisibility:
        """Project the whole OBB and test first-hit visible geometry, including the robot.

        Rays start at the RGB renderer's near plane. Transparent contact pads
        do not occlude, but visual meshes, fingers, held toys, and bin walls do.
        This is simulation ground truth, not an RGB perception algorithm.
        """
        e, c, config = self.engine, self.cameras[camera_name], self.config
        if name == "cleanup_bin":
            corners = self.bin_corners
            target_geoms = {e.model.geom(f"bin_{index}").id for index in range(5)}
        else:
            if name not in self.bounds:
                self.bounds[name] = body_bounding_box(e, name)
            item = read_object(e, name, self.bounds[name])
            corners = item.corners
            target_geoms = set(np.flatnonzero(e.model.geom_bodyid == e.model.body(name).id))
        camera = e.data.camera(camera_name)
        rotation = camera.xmat.reshape(3, 3) @ np.diag([1, -1, -1])
        points = (corners - camera.xpos) @ rotation
        near = e.model.vis.map.znear * e.model.stat.extent
        clipped = _clip_box_at_near_plane(points, near)
        if not len(clipped):
            return TargetVisibility(name, False, 0.0, 0.0, (0.0, 0.0), False, camera_name)
        focal = np.array([c.focal_x, c.focal_length])
        center = (np.array([c.width, c.height]) - 1) / 2
        pixels = clipped[:, :2] / clipped[:, 2:] * focal + center
        lower, upper = pixels.min(axis=0), pixels.max(axis=0)
        in_frame = bool(
            np.all(points[:, 2] >= near)
            and np.all(lower >= config.frame_margin)
            and np.all(upper <= np.array([c.width, c.height]) - 1 - config.frame_margin)
        )
        clipped_lower = np.maximum(lower, config.frame_margin)
        clipped_upper = np.minimum(upper, np.array([c.width, c.height]) - 1 - config.frame_margin)
        box_pixels = float(np.prod(np.maximum(0, clipped_upper - clipped_lower)))
        visible = 0
        grid = (np.arange(config.ray_grid_size) + 0.5) / config.ray_grid_size
        hit = np.array([-1], dtype=np.int32)
        if box_pixels > 0:
            for x in grid:
                for y in grid:
                    pixel = clipped_lower + [x, y] * (clipped_upper - clipped_lower)
                    direction = rotation @ np.r_[(pixel - center) / focal, 1.0]
                    origin = camera.xpos + direction * near
                    mujoco.mj_ray(e.model, e.data, origin, direction, None, True, -1, hit)
                    if hit[0] in target_geoms:
                        visible += 1
        fraction = visible / config.ray_grid_size**2
        # A close bin surrounds the gripper and can extend beyond the image.
        # Require a measured patch of bin, rather than every outer box corner.
        minimum = (
            config.min_bin_visible_fraction
            if name == "cleanup_bin"
            else config.min_visible_fraction
        )
        if held:
            minimum = config.min_held_visible_fraction
        minimum_pixels = config.min_held_pixels if held else config.min_box_pixels
        return TargetVisibility(
            name,
            in_frame,
            box_pixels,
            fraction,
            tuple((clipped_lower + clipped_upper) / 2),
            box_pixels * fraction >= minimum_pixels and fraction >= minimum,
            camera_name,
        )

    def best_view(
        self, name: str, *, held: bool = False, require_in_frame: bool = False
    ) -> TargetVisibility:
        """Report the strongest visibility evidence among the recorded RGB views."""
        views = [self.measure(name, held=held, camera_name=camera) for camera in self.cameras]
        selected = max(
            views,
            key=lambda view: (
                view.visible and (view.in_frame or not require_in_frame),
                view.box_pixels * view.visible_fraction,
            ),
        )
        if not require_in_frame or selected.in_frame or not selected.visible:
            return selected
        # A nearby box can straddle two base images. Preserve the individual
        # camera's projection and separately verify coverage of all eight OBB
        # corners across views with enough visible target geometry.
        corners = (
            self.bin_corners
            if name == "cleanup_bin"
            else read_object(self.engine, name, self.bounds[name]).corners
        )
        covered = np.zeros(len(corners), dtype=bool)
        covering_cameras = []
        e = self.engine
        near = e.model.vis.map.znear * e.model.stat.extent
        margin = self.config.frame_margin
        for view in views:
            if not view.visible:
                continue
            c = self.cameras[view.camera]
            camera = e.data.camera(view.camera)
            rotation = camera.xmat.reshape(3, 3) @ np.diag([1, -1, -1])
            points = (corners - camera.xpos) @ rotation
            pixels = (
                points[:, :2] / np.maximum(points[:, 2:], near) * [c.focal_x, c.focal_length]
                + (np.array([c.width, c.height]) - 1) / 2
            )
            inside = (
                (points[:, 2] > near)
                & np.all(pixels >= margin, axis=1)
                & np.all(pixels <= np.array([c.width, c.height]) - 1 - margin, axis=1)
            )
            if np.any(inside & ~covered):
                covering_cameras.append(view.camera)
                covered |= inside
        return (
            replace(selected, covering_cameras=tuple(covering_cameras))
            if covered.all()
            else selected
        )
