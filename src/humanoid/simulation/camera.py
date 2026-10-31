"""Render simulated RGB for training observations, never expert inputs."""

from io import BytesIO

import mujoco
import numpy as np
import pinocchio as pin
from PIL import Image

from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.camera import CameraFrame, RGBCameraConfig
from humanoid.types.collection import SimulationCameraObservation


def add_camera(spec: mujoco.MjSpec, name: str, camera: RGBCameraConfig) -> None:
    """Mount an RGB sensor and its physical housing before model compilation."""
    optical_rotation = np.asarray(camera.rotation)
    render_rotation = optical_rotation @ np.diag([1, -1, -1])
    parent = spec.worldbody if camera.parent_frame is None else spec.body(camera.parent_frame)
    if parent is None:
        raise ValueError(f"Camera parent frame does not exist: {camera.parent_frame}")
    # A separate fixed body adds the assembly's mass/inertia without replacing
    # the URDF link's explicit inertia. MuJoCo can fuse it at compilation.
    prefix = f"camera_{name}"
    parent = parent.add_body(name=f"{prefix}_assembly")
    sensor = parent.add_camera(
        name=name,
        pos=camera.position,
        xyaxes=np.r_[render_rotation[:, 0], render_rotation[:, 1]],
        fovy=camera.vertical_fov_degrees,
    )
    if camera.horizontal_fov_degrees is not None:
        sensor.resolution = [camera.width, camera.height]
        sensor.sensor_size = [camera.width, camera.height]
        sensor.focal_pixel = [camera.focal_x, camera.focal_length]
    if camera.mount is not None:
        mount = camera.mount
        axes = np.asarray(mount.rotation)
        parent.add_geom(
            name=f"{prefix}_bracket",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=mount.position,
            xyaxes=np.r_[axes[:, 0], axes[:, 1]],
            size=np.asarray(mount.size) / 2,
            mass=mount.mass,
            rgba=[0.32, 0.33, 0.35, 1],
            contype=1,
            conaffinity=2,
        )
        for index, hole in enumerate(mount.holes):
            parent.add_geom(
                name=f"{prefix}_screw_{index}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=np.asarray(hole) + axes[:, 1] * mount.screw_length / 2,
                xyaxes=np.r_[axes[:, 0], -axes[:, 2]],
                size=[mount.screw_radius, mount.screw_length / 2, 0],
                mass=0,
                rgba=[0.65, 0.67, 0.70, 1],
                contype=1,
                conaffinity=2,
            )
    if camera.housing_size is not None:
        half = np.asarray(camera.housing_size) / 2
        parent.add_geom(
            name=f"{prefix}_housing",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=np.asarray(camera.position) - optical_rotation[:, 2] * half[2],
            xyaxes=np.r_[optical_rotation[:, 0], optical_rotation[:, 1]],
            size=half,
            mass=camera.housing_mass,
            rgba=[0.18, 0.19, 0.21, 1],
            contype=1,
            conaffinity=2,
        )
        parent.add_geom(
            name=f"{prefix}_lens",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=camera.position,
            xyaxes=np.r_[optical_rotation[:, 0], optical_rotation[:, 1]],
            size=[0.004, 0.0005, 0],
            mass=0,
            rgba=[0.025, 0.06, 0.09, 1],
            contype=1,
            conaffinity=2,
        )
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, camera.width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, camera.height)


class SimulationCamera:
    def __init__(self, engine: NativeMujocoEngine, name: str, config: RGBCameraConfig):
        self.engine, self.name, self.config = engine, name, config
        self.renderer = mujoco.Renderer(engine.model, height=config.height, width=config.width)

    def capture(self, timestamp: float) -> SimulationCameraObservation:
        self.renderer.update_scene(self.engine.data, camera=self.name)
        rgb = self.renderer.render().copy()
        rgb_buffer = BytesIO()
        Image.fromarray(rgb).save(rgb_buffer, format="JPEG", quality=90)
        camera = self.engine.data.camera(self.name)
        optical = pin.SE3(camera.xmat.reshape(3, 3) @ np.diag([1, -1, -1]), camera.xpos.copy())
        intrinsics = np.array(
            [
                [self.config.focal_x, 0, (self.config.width - 1) / 2],
                [0, self.config.focal_length, (self.config.height - 1) / 2],
                [0, 0, 1],
            ]
        )
        return SimulationCameraObservation(
            CameraFrame(
                timestamp,
                self.name,
                f"{self.name}_optical",
                self.config.width,
                self.config.height,
                intrinsics,
                rgb_buffer.getvalue(),
            ),
            optical,
        )

    def close(self) -> None:
        self.renderer.close()
