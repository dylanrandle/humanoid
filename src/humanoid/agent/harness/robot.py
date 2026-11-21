"""Build a model observation from measured Triskel state and its live RGB camera."""

import base64
import io

import numpy as np
import pinocchio as pin
from PIL import Image

from humanoid.agent.client import AgentControlClient
from humanoid.config.camera import REALSENSE_CAMERA_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.robots.base import Robot
from humanoid.types.orchestrator import Mode


def pose_record(pose: pin.SE3) -> dict:
    return {
        "position_m": pose.translation.tolist(),
        "quaternion_xyzw": pin.Quaternion(pose.rotation).coeffs().tolist(),
    }


class TriskelTools:
    def __init__(self, client: AgentControlClient) -> None:
        self.client = client
        self.robot = Robot(TRISKEL_CONFIG)

    def observe(self, mode: Mode) -> tuple[dict, str]:
        rgb = self.client.observe().rgb
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
        status = {
            "mode": mode,
            "base_pose_world": pose_record(base),
            "tool_frame": self.robot.config.tool.frame,
            "tool_pose_base": pose_record(self.robot.get_tool_command_pose(q)),
            "gripper_positions_rad": q[self.robot.get_gripper_position_indices()].tolist(),
            "gripper_limits_rad": self.robot.get_gripper_limits(),
            "rgb_frame": rgb.frame_number,
        }
        buffer = io.BytesIO()
        Image.fromarray(rgb.data).save(buffer, format="JPEG", quality=90)
        return status, "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
