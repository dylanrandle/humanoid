"""Trusted realtime session; agent programs see only the MCP tool allowlist."""

from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

from PIL import Image

from humanoid.integrations.rax.tools import simulation_tools, validate_request
from humanoid.remote_control.observation import measured_geometry, optical_to_base
from humanoid.types.autonomy import EpisodeLifecycle, RobotExecution

if TYPE_CHECKING:
    from humanoid.robots.base import Robot
    from humanoid.types.robot import RobotState


class RealtimeSession:
    """Robot calls use the control API; task lifecycle stays with the operator service."""

    def __init__(
        self,
        lifecycle: EpisodeLifecycle,
        task: str = "toy_car_cleanup",
    ):
        self.task_name = task
        self.lifecycle = lifecycle
        self.execution: RobotExecution | None = None
        self.episode_id: str | None = None

    def start(self, *, formal: bool, seed: int) -> tuple[dict, list[dict]]:
        if type(formal) is not bool or type(seed) is not int or seed < 0:
            raise ValueError("Expected a boolean formal flag and a nonnegative integer seed.")
        self.close()
        self.episode_id = self.lifecycle.start_episode(formal=formal, seed=seed)
        try:
            self.execution = self.lifecycle.create_execution()
            self.execution.wait_until_ready()
        except BaseException:
            self.close()
            raise
        return self.call("humanoid_status", {})

    def tools(self) -> list[dict]:
        return simulation_tools()

    def status(self) -> dict:
        execution = self.execution
        if execution is None:
            return {"active": False, "task": self.task_name, "episode_id": self.episode_id}
        state = execution.state()
        robot = execution.robot
        geometry = measured_geometry(robot, state)
        return {
            "active": True,
            "runtime": "sim",
            "task": self.task_name,
            "episode_id": self.episode_id,
            "step_id": execution.step_id,
            "limits": asdict(execution.config),
            "tool_velocity_limits": geometry["tool_velocity_limits"],
            "base_velocity_limits": (
                asdict(robot.config.base.velocity_limits) if robot.config.base else None
            ),
            "tool_frame": geometry["tool_frame"],
            "gripper_limits": geometry["gripper_limits"],
            "state": {
                "timestamp": state.timestamp,
                "joint_positions": state.joint_positions.tolist(),
                "gripper_positions": geometry["gripper_positions"],
                "joint_velocities": state.joint_velocities.tolist(),
                "arm_joints": self._arm_joint_state(robot, state),
                "tool_pose_base": geometry["tool_pose_base"],
                "base_pose_world": geometry["base_pose_world"],
            },
            "localization": "robot_state_base_pose",
            "camera_frames": "native_unaligned",
        }

    @staticmethod
    def _arm_joint_state(robot: Robot, state: RobotState) -> dict:
        """Decode named actuator feedback; raw Pinocchio q is not an angle list."""
        joints = {}
        for index in robot.get_arm_joint_indices():
            joint = robot.model.joints[index + 1]
            lower = float(robot.model.lowerPositionLimit[joint.idx_q])
            upper = float(robot.model.upperPositionLimit[joint.idx_q])
            joints[robot.model.names[index + 1]] = {
                "position": robot.joint_position_from_q(state.joint_positions, index),
                "velocity": float(state.joint_velocities[joint.idx_v]),
                # Continuous revolute joints encode cos/sin in q; those bounds
                # describe the encoding, not physical angular limits.
                "lower_position_limit": lower if joint.nq == 1 and math.isfinite(lower) else None,
                "upper_position_limit": upper if joint.nq == 1 and math.isfinite(upper) else None,
            }
        return joints

    def call(self, name: str, arguments: dict) -> tuple[dict, list[dict]]:
        validate_request(name, arguments)
        if name == "humanoid_status":
            return self.status(), []
        if self.execution is None:
            raise RuntimeError("Start an episode before using robot tools.")
        if name == "humanoid_observe":
            return self._observe()
        result = self.execution.execute(name.removeprefix("humanoid_"), arguments)
        return {**self.status(), "motion": result}, []

    def _observe(self) -> tuple[dict, list[dict]]:
        assert self.execution is not None
        frames = self.execution.observe()
        state = self.execution.state()
        robot = self.execution.robot
        images, metadata = [], []
        for name, frame in (("rgb", frames.rgb), ("depth", frames.depth)):
            output = io.BytesIO()
            Image.fromarray(frame.data).save(output, format="PNG")
            images.append(
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": base64.b64encode(output.getvalue()).decode("ascii"),
                }
            )
            metadata.append(
                {
                    "kind": name,
                    "content_index": len(images),
                    "frame_id": frame.frame_id,
                    "intrinsics": asdict(frame.intrinsics),
                    "optical_to_base": optical_to_base(robot, state, frame.frame_id),
                    "encoding": frame.encoding.value,
                    "depth_scale": frame.depth_scale,
                    "timestamp": frame.timestamp,
                }
            )
        return {**self.status(), "attachments": metadata}, images

    def evaluate(self) -> dict:
        evaluation = self.lifecycle.evaluate_episode()
        return {
            "status": "completed",
            "task": self.task_name,
            "episode_id": self.episode_id,
            "step_id": None if self.execution is None else self.execution.step_id,
            "task_complete": evaluation.task_complete,
            "evaluation_kind": "binary_task_completion",
            "decided_at_unix_s": time.time(),
        }

    def close(self) -> None:
        try:
            if self.execution is not None:
                self.execution.close()
        finally:
            self.execution = None
            if self.episode_id is not None:
                self.episode_id = None
                self.lifecycle.close_episode()
