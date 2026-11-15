"""Acquire native RGB and depth without alignment or depth post-processing."""

import importlib
import sys
import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from humanoid.types.camera import (
    CameraEncoding,
    CameraFrames,
    CameraImage,
    CameraIntrinsics,
    RealSenseCameraConfig,
)


class RealSenseCamera:
    def __init__(
        self,
        config: RealSenseCameraConfig,
        *,
        sdk: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self._sdk = sdk
        self._clock = clock
        self._pipeline: Any = None
        self._intrinsics: dict[CameraEncoding, CameraIntrinsics] = {}
        self._depth_scale = 0.0
        self._serial_number = ""
        self._last_frame_at = 0.0

    def start(self) -> None:
        if self._pipeline is not None:
            raise RuntimeError("RealSense camera is already streaming.")
        if self._sdk is None:
            if sys.platform == "darwin":
                raise RuntimeError(
                    "The real camera driver is not supported on macOS. Use simulation "
                    "or run the real-hardware stack on the robot."
                )
            try:
                self._sdk = importlib.import_module("pyrealsense2")
            except ImportError as error:
                raise RuntimeError(
                    "RealSense SDK unavailable. Run `uv sync`; on Raspberry Pi/ARM, "
                    "install a matching SDK wheel as described in README.md."
                ) from error
        # The SDK exposes native extension attributes that its Python package does not type.
        sdk = cast(Any, self._sdk)
        settings = sdk.config()
        if self.config.serial_number is not None:
            settings.enable_device(self.config.serial_number)
        settings.enable_stream(
            sdk.stream.color,
            self.config.width,
            self.config.height,
            sdk.format.rgb8,
            self.config.fps,
        )
        settings.enable_stream(
            sdk.stream.depth, self.config.width, self.config.height, sdk.format.z16, self.config.fps
        )
        pipeline = sdk.pipeline()
        profile = pipeline.start(settings)
        self._pipeline = pipeline
        try:
            device = profile.get_device()
            self._depth_scale = float(device.first_depth_sensor().get_depth_scale())
            if not np.isfinite(self._depth_scale) or self._depth_scale <= 0:
                raise RuntimeError("RealSense reported an invalid depth scale.")
            self._serial_number = str(device.get_info(sdk.camera_info.serial_number))
            self._intrinsics = {
                encoding: self._read_intrinsics(profile.get_stream(stream))
                for encoding, stream in (
                    (CameraEncoding.RGB8, sdk.stream.color),
                    (CameraEncoding.DEPTH16, sdk.stream.depth),
                )
            }
            self._last_frame_at = self._clock()
        except BaseException:
            self.stop()
            raise

    @staticmethod
    def _read_intrinsics(profile: Any) -> CameraIntrinsics:
        intrinsic = profile.as_video_stream_profile().get_intrinsics()
        return CameraIntrinsics(
            width=int(intrinsic.width),
            height=int(intrinsic.height),
            fx=float(intrinsic.fx),
            fy=float(intrinsic.fy),
            cx=float(intrinsic.ppx),
            cy=float(intrinsic.ppy),
            distortion_model=str(intrinsic.model).removeprefix("distortion."),
            distortion_coefficients=tuple(float(x) for x in intrinsic.coeffs),
        )

    def read(self) -> CameraFrames | None:
        """Poll the latest frameset; fail after a bounded period without complete frames."""
        if self._pipeline is None:
            raise RuntimeError("RealSense camera is not streaming.")
        frames = self._pipeline.poll_for_frames()
        now = self._clock()
        if frames:
            rgb = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if rgb and depth:
                result = CameraFrames(
                    rgb=self._image(rgb, CameraEncoding.RGB8, now),
                    depth=self._image(depth, CameraEncoding.DEPTH16, now),
                )
                self._last_frame_at = now
                return result
        if now - self._last_frame_at >= self.config.frame_timeout_seconds:
            raise RuntimeError("RealSense timed out waiting for native RGB and depth frames.")
        return None

    def _image(self, frame: Any, encoding: CameraEncoding, timestamp: float) -> CameraImage:
        is_depth = encoding is CameraEncoding.DEPTH16
        return CameraImage(
            timestamp=timestamp,
            device_timestamp=float(frame.get_timestamp()) / 1000.0,
            timestamp_domain=str(frame.get_frame_timestamp_domain()),
            frame_number=int(frame.get_frame_number()),
            frame_id=self.config.depth_frame if is_depth else self.config.rgb_frame,
            serial_number=self._serial_number,
            encoding=encoding,
            # Own the storage: SDK buffers may be recycled after releasing the frameset.
            data=np.array(frame.get_data(), copy=True, order="C"),
            intrinsics=self._intrinsics[encoding],
            depth_scale=self._depth_scale if is_depth else 0.0,
        )

    def stop(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        self._intrinsics.clear()
        if pipeline is not None:
            pipeline.stop()
