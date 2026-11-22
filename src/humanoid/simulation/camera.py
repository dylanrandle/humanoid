"""Render native RGB and metric optical-axis depth from the URDF camera frames."""

import math
from dataclasses import replace

import mujoco
import numpy as np

from humanoid.types.camera import (
    CameraEncoding,
    CameraFrames,
    CameraImage,
    CameraIntrinsics,
    RegisteredDepth,
    SimulatedCameraConfig,
)


class FrameClock:
    """Select physics ticks on the sensor's simulation-time grid."""

    def __init__(self, fps: int):
        self.fps = fps
        self._last_frame = -1

    def due(self, simulation_time: float) -> bool:
        frame = math.floor(simulation_time * self.fps + 1e-9)
        if frame == self._last_frame:
            return False
        self._last_frame = frame
        return True


class MujocoCamera:
    def __init__(self, model: mujoco.MjModel, config: SimulatedCameraConfig) -> None:
        self.model = model
        self.config = config
        self._renderer: mujoco.Renderer | None = None
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[3] = 0  # URDF collision proxies.
        self.reset()

    def reset(self) -> None:
        self._clock = FrameClock(self.config.fps)
        self._frame_number = 0

    def read(self, data: mujoco.MjData, timestamp: float) -> CameraFrames | None:
        if not self._clock.due(data.time):
            return None
        return self.render(data, timestamp)

    def render(self, data: mujoco.MjData, timestamp: float) -> CameraFrames:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.config.height, width=self.config.width
            )
        renderer = self._renderer
        renderer.update_scene(data, camera=self.config.rgb_frame, scene_option=self._scene_option)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        try:
            registered_pixels = self._depth_units(renderer.render())
            renderer.update_scene(
                data, camera=self.config.depth_frame, scene_option=self._scene_option
            )
            depth = self._depth_units(renderer.render())
        finally:
            renderer.disable_depth_rendering()
        self._frame_number += 1
        color = self._image(rgb, CameraEncoding.RGB8, timestamp, data.time)
        native = self._image(depth, CameraEncoding.DEPTH16, timestamp, data.time)
        return CameraFrames(
            rgb=color,
            depth=native,
            registered=RegisteredDepth(
                camera_model=self.config.model,
                rgb_frame_number=color.frame_number,
                image=replace(
                    native,
                    frame_id=color.frame_id,
                    intrinsics=color.intrinsics,
                    data=registered_pixels,
                ),
            ),
        )

    def _depth_units(self, meters: np.ndarray) -> np.ndarray:
        far = self.model.vis.map.zfar * self.model.stat.extent
        units = np.rint(meters / self.config.depth_scale)
        valid = (
            np.isfinite(units)
            & (units > 0)
            & (units <= np.iinfo(np.uint16).max)
            & (meters < far * (1 - 1e-5))
        )
        return np.where(valid, units, 0).astype(np.uint16)

    def _image(
        self, pixels: np.ndarray, encoding: CameraEncoding, timestamp: float, simulation_time: float
    ) -> CameraImage:
        is_depth = encoding is CameraEncoding.DEPTH16
        frame = self.config.depth_frame if is_depth else self.config.rgb_frame
        camera = self.model.camera(frame)
        sensor_size = self.model.cam_sensorsize[camera.id]
        if np.all(sensor_size > 0):
            focal_lengths = self.model.cam_intrinsic[camera.id, :2]
            fx, fy = focal_lengths * [self.config.width, self.config.height] / sensor_size
        else:
            fovy = float(camera.fovy[0])
            fx = fy = self.config.height / (2 * math.tan(math.radians(fovy) / 2))
        return CameraImage(
            timestamp=timestamp,
            device_timestamp=simulation_time,
            timestamp_domain="simulation_clock",
            frame_number=self._frame_number,
            frame_id=frame,
            serial_number=f"simulation-{self.config.rgb_frame}",
            encoding=encoding,
            data=pixels,
            intrinsics=CameraIntrinsics(
                width=self.config.width,
                height=self.config.height,
                fx=float(fx),
                fy=float(fy),
                cx=(self.config.width - 1) / 2,
                cy=(self.config.height - 1) / 2,
                distortion_model="none",
                distortion_coefficients=(0.0,) * 5,
            ),
            depth_scale=self.config.depth_scale if is_depth else 0.0,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
