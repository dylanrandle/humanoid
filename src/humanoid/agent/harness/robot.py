"""Build a model observation from measured robot state and its live RGB camera."""

import base64
import io

import numpy as np
import pinocchio as pin
from PIL import Image

from humanoid.agent.harness.depth import DepthTools
from humanoid.agent.harness.observation import PinnedObservation
from humanoid.agent.harness.segmentation import SegmentationTools
from humanoid.perception.masks import MaskStore
from humanoid.robots.base import Robot
from humanoid.types.agent import HarnessRobot
from humanoid.types.camera import (
    CameraImage,
    CameraIntrinsicsSummary,
    CameraSummary,
    CameraView,
    OpticalCalibration,
    PoseRecord,
)
from humanoid.types.observation import ObservationSummary
from humanoid.types.orchestrator import Mode


def image_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def pose_record(pose: pin.SE3) -> PoseRecord:
    return {
        "position_m": pose.translation.tolist(),
        "quaternion_xyzw": pin.Quaternion(pose.rotation).coeffs().tolist(),
    }


class RobotTools:
    def __init__(self, client: HarnessRobot, robot: Robot) -> None:
        self.client = client
        self.robot = robot
        self.observation = PinnedObservation(MaskStore())
        self.segmentation = SegmentationTools(self.observation)
        self.depth = DepthTools(self.observation)

    def close(self) -> None:
        try:
            self.depth.close()
        finally:
            self.segmentation.close()

    def _calibration(
        self, frame: CameraImage, joints: np.ndarray, base: pin.SE3 | None
    ) -> OpticalCalibration:
        pose = self.robot.get_frame_pose(frame.frame_id, joints)
        intrinsics = frame.intrinsics
        return OpticalCalibration(
            optical_frame=frame.frame_id,
            camera_pose=pose_record(base.inverse() * pose if base is not None else pose),
            intrinsics=CameraIntrinsicsSummary(
                width=intrinsics.width,
                height=intrinsics.height,
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                cx=intrinsics.cx,
                cy=intrinsics.cy,
                distortion_model=intrinsics.distortion_model,
                distortion_coefficients=intrinsics.distortion_coefficients,
            ),
        )

    def capture(self, mode: Mode) -> tuple[ObservationSummary, dict[str, CameraView]]:
        captures = self.client.capture()
        primary = next(iter(captures))
        measured, _ = captures[primary]
        cameras: dict[str, CameraSummary] = {}
        for name, (sample, view) in captures.items():
            mount = self.robot.config.cameras[name]
            expected = {c.rgb_frame for c in (mount.hardware, mount.simulation) if c}
            if view.rgb.frame_id not in expected:
                raise ValueError(
                    f"Expected {name} camera frame {expected}, got {view.rgb.frame_id!r}."
                )
            if sample.mode is not mode:
                raise RuntimeError("Robot mode changed during the task.")
            joints = sample.robot_state.joint_positions
            if joints.shape != (self.robot.model.nq,) or not np.isfinite(joints).all():
                raise ValueError(
                    f"Joint feedback does not match the {self.robot.config.name} model."
                )
            base = self.robot.get_base_pose(joints)
            cameras[name] = CameraSummary(
                **self._calibration(view.rgb, joints, base),
                model=view.registered.camera_model if view.registered else None,
                observation_id=self.observation.pin(view, additional=bool(cameras)),
                frame_number=view.rgb.frame_number,
                device_timestamp=view.rgb.device_timestamp,
                timestamp_domain=view.rgb.timestamp_domain,
                raw_depth=self._calibration(view.depth, joints, base)
                if view.depth is not None
                else None,
                registered_depth_available=view.registered is not None,
            )
        q = measured.robot_state.joint_positions
        base = self.robot.get_base_pose(q)
        status: ObservationSummary = {
            "mode": mode,
            "cameras": cameras,
            "base_pose_world": pose_record(base) if base is not None else None,
            "tool_frame": self.robot.config.tool.frame,
            "tool_pose_reference_frame": self.robot.config.base.frame
            if self.robot.config.base
            else "world",
            "tool_pose": pose_record(self.robot.get_tool_command_pose(q)),
            "gripper_units": list(self.robot.config.gripper.units)
            if self.robot.config.gripper
            else [],
            "gripper_positions": q[self.robot.get_gripper_position_indices()].tolist(),
            "gripper_limits": self.robot.get_gripper_limits(),
            "primary_camera": primary,
        }
        return status, {name: view for name, (_, view) in captures.items()}

    def observe(self, mode: Mode) -> tuple[ObservationSummary, dict[str, str]]:
        status, views = self.capture(mode)
        images = {
            f"{name} RGB": image_url(Image.fromarray(view.rgb.data)) for name, view in views.items()
        }
        return status, images
