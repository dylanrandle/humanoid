"""Build a model observation from measured Triskel state and its live RGB camera."""

import base64
import io

import numpy as np
import pinocchio as pin
from PIL import Image

from humanoid.agent.client import AgentControlClient
from humanoid.agent.harness.depth import DepthTools
from humanoid.agent.harness.observation import PinnedObservation
from humanoid.agent.harness.segmentation import SegmentationTools
from humanoid.config.camera import CAMERA_PREVIEW_CONFIG, REALSENSE_CAMERA_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.perception.depth import colorize_depth
from humanoid.perception.masks import MaskStore
from humanoid.robots.base import Robot
from humanoid.types.agent_decision import ObservationStatus, PoseRecord
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


class TriskelTools:
    def __init__(self, client: AgentControlClient) -> None:
        self.client = client
        self.robot = Robot(TRISKEL_CONFIG)
        self.observation = PinnedObservation(MaskStore())
        self.segmentation = SegmentationTools(self.observation)
        self.depth = DepthTools(self.observation)

    def close(self):
        try:
            self.depth.close()
        finally:
            self.segmentation.close()

    def observe(self, mode: Mode) -> tuple[ObservationStatus, str, str | None]:
        frames = self.client.observe()
        rgb = frames.rgb
        if rgb.frame_id != REALSENSE_CAMERA_CONFIG.rgb_frame:
            raise ValueError("Camera frame does not match the Triskel model.")
        measured = self.client.observation()
        if measured.mode is not mode:
            raise RuntimeError("Robot mode changed during the task.")
        q = measured.robot_state.joint_positions
        if q.shape != (self.robot.model.nq,) or not np.isfinite(q).all():
            raise ValueError("Joint feedback does not match the Triskel model.")
        base = self.robot.get_base_pose(q)
        assert base is not None
        status: ObservationStatus = {
            "mode": mode,
            "base_pose_world": pose_record(base),
            "tool_frame": self.robot.config.tool.frame,
            "tool_pose_base": pose_record(self.robot.get_tool_command_pose(q)),
            "gripper_positions_rad": q[self.robot.get_gripper_position_indices()].tolist(),
            "gripper_limits_rad": self.robot.get_gripper_limits(),
            "rgb_frame": rgb.frame_number,
            "observation_id": self.observation.pin(frames),
            "rgb_size_xy": [rgb.intrinsics.width, rgb.intrinsics.height],
            "registered_depth_available": frames.registered is not None,
        }
        depth_image = None
        if frames.depth is not None:
            near = CAMERA_PREVIEW_CONFIG.depth_near_meters
            far = CAMERA_PREVIEW_CONFIG.default_depth_far_meters
            metres = frames.depth.data.astype(np.float32) * frames.depth.depth_scale
            depth_image = image_url(Image.fromarray(colorize_depth(metres, near, far)))
            status["measured_depth_preview_range_m"] = [near, far]
        return status, image_url(Image.fromarray(rgb.data)), depth_image
