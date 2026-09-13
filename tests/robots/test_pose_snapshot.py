"""Tests for the measured-pose snapshot utility."""

import json
from unittest.mock import MagicMock

import numpy as np
import pinocchio as pin
import pytest

import humanoid.robots.utils.pose_snapshot as snapshot_module
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.constants import Topic
from humanoid.robots.base import Robot
from humanoid.types.actuator import ActuatorControlMode
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import RobotState

TASK_POSITION_SIZE = 3
TASK_QUATERNION_SIZE = 4


def _home_state(robot: Robot) -> RobotState:
    return RobotState(
        timestamp=12.5,
        joint_positions=TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy(),
        joint_velocities=np.zeros(robot.model.nv),
        actuator_temperatures=np.empty(0),
    )


def test_pose_snapshot_contains_named_position_joints_and_tool_command_pose():
    robot = Robot(TRISKEL_CONFIG)
    state = _home_state(robot)

    snapshot = snapshot_module.pose_snapshot_from_state(robot, state)

    expected_joint_names = [
        name
        for name in robot.actuator_joint_names
        if TRISKEL_CONFIG.actuator_control_modes[name] is ActuatorControlMode.POSITION
    ]
    assert list(snapshot.joint_positions_rad) == expected_joint_names
    assert "front_left_wheel" not in snapshot.joint_positions_rad
    assert snapshot.task_frame == TRISKEL_CONFIG.base.frame  # type: ignore[union-attr]
    expected_pose = robot.get_tool_command_pose(state.joint_positions)
    np.testing.assert_allclose(snapshot.task_position_m, expected_pose.translation)
    reported_quaternion = pin.Quaternion(*snapshot.task_quaternion_wxyz)
    np.testing.assert_allclose(reported_quaternion.toRotationMatrix(), expected_pose.rotation)


def test_pose_snapshot_format_is_copyable_json():
    robot = Robot(TRISKEL_CONFIG)
    snapshot = snapshot_module.pose_snapshot_from_state(robot, _home_state(robot))

    payload = json.loads(snapshot_module.format_pose_snapshot(snapshot, label="start"))

    assert payload["label"] == "start"
    assert payload["robot"] == "triskel"
    assert payload["state_timestamp_s"] == pytest.approx(12.5)
    assert len(payload["task_pose"]["position_m"]) == TASK_POSITION_SIZE
    assert len(payload["task_pose"]["quaternion_wxyz"]) == TASK_QUATERNION_SIZE


def test_capture_pose_snapshot_reads_once_and_closes(monkeypatch):
    robot = Robot(TRISKEL_CONFIG)
    state = _home_state(robot)
    subscriber = MagicMock()
    subscriber.receive.return_value = state
    monkeypatch.setattr(snapshot_module, "Subscriber", lambda **_kwargs: subscriber)

    snapshot = snapshot_module.capture_pose_snapshot(TRISKEL_CONFIG, timeout_s=0.25)

    assert snapshot.state_timestamp_s == state.timestamp
    subscriber.receive.assert_called_once_with(Topic.ROBOT_STATE, timeout=250)
    subscriber.close.assert_called_once_with()


def test_capture_pose_snapshot_reports_timeout_and_closes(monkeypatch):
    subscriber = MagicMock()
    subscriber.receive.return_value = None
    monkeypatch.setattr(snapshot_module, "Subscriber", lambda **_kwargs: subscriber)

    with pytest.raises(RuntimeError, match=r"No robot state received within 0\.25 seconds"):
        snapshot_module.capture_pose_snapshot(TRISKEL_CONFIG, timeout_s=0.25)

    subscriber.close.assert_called_once_with()


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, np.inf, np.nan])
def test_capture_pose_snapshot_rejects_invalid_timeout(timeout_s):
    with pytest.raises(ValueError, match="timeout"):
        snapshot_module.capture_pose_snapshot(TRISKEL_CONFIG, timeout_s=timeout_s)
