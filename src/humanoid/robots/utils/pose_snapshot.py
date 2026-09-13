"""Print the robot's measured joint and tool pose for tracking-test setup."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Any

import pinocchio as pin

from humanoid.config import ROBOT_CONFIG, ROBOT_CONFIGS
from humanoid.constants import Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.robots.base import Robot
from humanoid.types.actuator import ActuatorControlMode
from humanoid.types.robot import RobotConfig, RobotName, RobotState

DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, kw_only=True)
class PoseSnapshot:
    """Measured position-controlled joints and tool-command pose."""

    robot_name: str
    state_timestamp_s: float
    joint_positions_rad: dict[str, float]
    task_frame: str
    task_position_m: tuple[float, float, float]
    task_quaternion_wxyz: tuple[float, float, float, float]


def pose_snapshot_from_state(robot: Robot, state: RobotState) -> PoseSnapshot:
    """Convert one measured robot state into copyable joint/task coordinates."""
    expected_shape = (robot.model.nq,)
    if state.joint_positions.shape != expected_shape:
        raise ValueError(
            f"Robot state has {len(state.joint_positions)} positions, "
            f"but {robot.config.name.value} expects {robot.model.nq}."
        )

    joint_positions_rad = {
        joint_name: robot.joint_position_from_q(
            state.joint_positions,
            robot.joint_name_to_idx(joint_name),
        )
        for joint_name in robot.actuator_joint_names
        if robot.config.actuator_control_modes[joint_name] is ActuatorControlMode.POSITION
    }
    task_pose = robot.get_tool_command_pose(state.joint_positions)
    quaternion = pin.Quaternion(task_pose.rotation)
    quaternion.normalize()
    quaternion_wxyz = (quaternion.w, quaternion.x, quaternion.y, quaternion.z)
    if quaternion_wxyz[0] < 0.0:
        quaternion_wxyz = tuple(-component for component in quaternion_wxyz)

    task_frame = robot.config.base.frame if robot.config.base is not None else "world"
    return PoseSnapshot(
        robot_name=robot.config.name.value,
        state_timestamp_s=float(state.timestamp),
        joint_positions_rad=joint_positions_rad,
        task_frame=task_frame,
        task_position_m=(
            float(task_pose.translation[0]),
            float(task_pose.translation[1]),
            float(task_pose.translation[2]),
        ),
        task_quaternion_wxyz=quaternion_wxyz,
    )


def capture_pose_snapshot(
    robot_config: RobotConfig = ROBOT_CONFIG,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
) -> PoseSnapshot:
    """Wait for one live robot-state sample and snapshot its measured pose."""
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout must be positive and finite")

    robot = Robot(robot_config)
    subscriber = Subscriber(topics=[Topic.ROBOT_STATE])
    try:
        state = subscriber.receive(Topic.ROBOT_STATE, timeout=math.ceil(timeout_s * 1_000.0))
    finally:
        subscriber.close()
    if state is None:
        raise RuntimeError(f"No robot state received within {timeout_s:g} seconds")
    return pose_snapshot_from_state(robot, state)


def format_pose_snapshot(snapshot: PoseSnapshot, label: str | None = None) -> str:
    """Format a snapshot as JSON that can be pasted into an issue or chat."""
    payload: dict[str, Any] = {
        "robot": snapshot.robot_name,
        "state_timestamp_s": snapshot.state_timestamp_s,
        "joint_positions_rad": snapshot.joint_positions_rad,
        "task_pose": {
            "frame": snapshot.task_frame,
            "position_m": snapshot.task_position_m,
            "quaternion_wxyz": snapshot.task_quaternion_wxyz,
        },
    }
    if label is not None:
        payload = {"label": label, **payload}
    return json.dumps(payload, indent=2, allow_nan=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Snapshot measured joint angles and the configured tool-command pose."
    )
    parser.add_argument(
        "--robot",
        type=RobotName,
        choices=list(RobotName),
        default=ROBOT_CONFIG.name,
        help=f"Robot model (default: {ROBOT_CONFIG.name.value}).",
    )
    parser.add_argument(
        "--label",
        help="Optional label included in the output, such as 'start' or 'end'.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Seconds to wait for a robot-state sample (default: {DEFAULT_TIMEOUT_SECONDS:g}).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        snapshot = capture_pose_snapshot(ROBOT_CONFIGS[args.robot], args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        print(f"Pose snapshot failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(format_pose_snapshot(snapshot, args.label))


if __name__ == "__main__":
    main()
