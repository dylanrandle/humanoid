import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.remote_control import get_remote_control_policy_config
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.integrations.rax.backend import HumanoidBackend
from humanoid.integrations.rax.benchmark import validate_settings
from humanoid.integrations.rax.server import dispatch
from humanoid.integrations.rax.session import RealtimeSession
from humanoid.integrations.rax.tools import tool_definitions
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import RobotState
from humanoid.types.tasks import TaskEvaluation


def test_agent_tool_catalog_has_no_privileged_methods():
    assert {t["name"] for t in tool_definitions()} == {
        "humanoid_status",
        "humanoid_observe",
        "humanoid_move_base",
        "humanoid_move_tool",
        "humanoid_gripper",
        "humanoid_hold",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"method": "close", "arguments": {"extra": 1}},
        {"method": "call", "arguments": {"name": "reset", "arguments": {}}},
        {"method": "start", "arguments": {"task": "unknown", "seed": 1, "formal": False}},
    ],
)
def test_rpc_rejects_unknown_operations(payload):
    session = RealtimeSession(Mock())
    with pytest.raises(ValueError):
        dispatch(session, payload)


def test_evaluation_delegates_to_operator_without_exposing_oracle_details():
    lifecycle = Mock()
    lifecycle.evaluate_episode.return_value = TaskEvaluation(
        task_complete=False, objects_delivered=1, objects_total=3, reason="cars_remaining"
    )
    result = RealtimeSession(lifecycle).evaluate()
    assert result["task_complete"] is False
    assert "objects_delivered" not in result
    assert "objects_total" not in result
    lifecycle.evaluate_episode.assert_called_once()


def test_start_delegates_to_ui_and_creates_only_realtime_client(monkeypatch):
    lifecycle = Mock()
    lifecycle.start_episode.return_value = "episode-1"
    execution = Mock()
    lifecycle.create_execution.return_value = execution
    session = RealtimeSession(lifecycle)
    monkeypatch.setattr(session, "status", lambda: {"active": True})
    status, images = session.start(formal=True, seed=3)
    lifecycle.start_episode.assert_called_once_with(formal=True, seed=3)
    lifecycle.create_execution.assert_called_once()
    execution.wait_until_ready.assert_called_once()
    assert status["active"] and not images
    session.close()
    lifecycle.close_episode.assert_called_once()


def test_status_exposes_named_measured_arm_joints_and_urdf_limits():
    robot = Robot(TRISKEL_CONFIG)
    q = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy()
    wrist = robot.model.getJointId("arm_5")
    joint = robot.model.joints[wrist]
    q[joint.idx_q] = robot.model.upperPositionLimit[joint.idx_q] - 0.005
    velocity = np.arange(robot.model.nv, dtype=float) / 10
    state = RobotState(
        timestamp=1.0,
        joint_positions=q,
        joint_velocities=velocity,
        actuator_temperatures=np.zeros(robot.model.nq),
    )
    execution = Mock(robot=robot, config=get_remote_control_policy_config(), step_id=0)
    execution.state.return_value = state
    session = RealtimeSession(Mock())
    session.execution = execution

    status, images = session.call("humanoid_status", {})

    assert not images
    joints = status["state"]["arm_joints"]
    assert set(joints) == {f"arm_{index}" for index in range(1, 8)}
    assert joints["arm_5"] == {
        "position": pytest.approx(2.6948),
        "velocity": velocity[joint.idx_v],
        "lower_position_limit": pytest.approx(-3.1907),
        "upper_position_limit": pytest.approx(2.6998),
    }
    json.dumps(status, allow_nan=False)


def test_named_continuous_arm_feedback_decodes_angle_without_cosine_sine_limits():
    model = pin.Model()
    model.addJoint(0, pin.JointModelRUBZ(), pin.SE3.Identity(), "continuous_arm")
    robot = Mock(spec=Robot, model=model)
    robot.get_arm_joint_indices.return_value = [0]
    robot.joint_position_from_q.side_effect = lambda q, i: Robot.joint_position_from_q(robot, q, i)
    state = RobotState(
        timestamp=1.0,
        joint_positions=np.array([np.cos(2.4), np.sin(2.4)]),
        joint_velocities=np.array([-0.3]),
        actuator_temperatures=np.zeros(2),
    )

    joints = RealtimeSession._arm_joint_state(robot, state)

    assert joints == {
        "continuous_arm": {
            "position": pytest.approx(2.4),
            "velocity": -0.3,
            "lower_position_limit": None,
            "upper_position_limit": None,
        }
    }
    json.dumps(joints, allow_nan=False)


def test_stdlib_adapter_lifecycle_dispatches_to_trusted_service(monkeypatch):
    monkeypatch.setenv("HUMANOID_RAX_TOKEN", "test-token")
    backend = HumanoidBackend(SimpleNamespace(task="toy_car_cleanup"))
    request = Mock(return_value={"value": {"active": True}, "images": []})
    monkeypatch.setattr(backend, "_request", request)
    assert backend.start(formal=False, seed=12)[0]["active"]
    request.assert_called_with("start", task="toy_car_cleanup", formal=False, seed=12)
    backend.call("humanoid_hold", {"duration": 0.02})
    request.assert_called_with("call", name="humanoid_hold", arguments={"duration": 0.02})
    backend.close()
    request.assert_called_with("close")


@pytest.mark.parametrize(
    "values", [{"formal_workers": 2}, {"exploration_envs": 2}, {"observation_profile": "rgb-only"}]
)
def test_registry_rejects_unsupported_parallel_or_observation_modes(values):
    with pytest.raises(ValueError):
        validate_settings(values)
