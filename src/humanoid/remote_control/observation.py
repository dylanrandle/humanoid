"""Measured kinematics shared by remote observation clients."""

from dataclasses import asdict

import pinocchio as pin

from humanoid.robots.base import Robot
from humanoid.types.robot import RobotState


def pose_record(pose: pin.SE3) -> dict:
    return {
        "position": pose.translation.tolist(),
        "quaternion_xyzw": pin.Quaternion(pose.rotation).coeffs().tolist(),
    }


def optical_to_base(robot: Robot, state: RobotState, frame_id: str) -> dict:
    pose = robot.get_frame_pose(frame_id, state.joint_positions).copy()
    base = robot.get_base_pose(state.joint_positions)
    return pose_record(pose if base is None else base.inverse() * pose)


def measured_geometry(robot: Robot, state: RobotState) -> dict:
    positions = state.joint_positions
    tool = pose_record(robot.get_tool_command_pose(positions))
    base = robot.get_base_pose(positions)
    return {
        "tool_frame": robot.config.tool.frame,
        "tool_velocity_limits": asdict(robot.config.tool.velocity_limits),
        "gripper_limits": robot.get_gripper_limits(),
        "gripper_positions": positions[robot.get_gripper_position_indices()].tolist(),
        "tool_pose_base": tool,
        "base_pose_world": None if base is None else pose_record(base),
    }
