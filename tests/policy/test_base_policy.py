"""Tests for the Policy base class's mode-gated execution."""

import numpy as np

from humanoid.policy.base import Policy
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotState


class _RecordingPolicy(Policy):
    """Policy stub that records calls to step() and reset()."""

    mode = Mode.HOMING

    def __init__(self) -> None:
        self.step_calls = 0
        self.reset_calls = 0

    def step(self, observation: Observation) -> Action:
        self.step_calls += 1
        return Action(joint_positions=np.zeros(3))

    def reset(self) -> None:
        self.reset_calls += 1


def _obs(mode: Mode | None) -> Observation:
    state = RobotState(
        timestamp=0.0,
        joint_positions=np.zeros(3),
        joint_velocities=np.zeros(3),
        actuator_temperatures=np.zeros(3),
    )
    return Observation(robot_state=state, mode=mode)


def test_matching_mode_runs_step():
    policy = _RecordingPolicy()
    action = policy(_obs(Mode.HOMING))

    assert policy.step_calls == 1
    assert policy.reset_calls == 0
    assert action.joint_positions is not None


def test_mismatched_mode_resets_and_returns_empty_action():
    policy = _RecordingPolicy()
    action = policy(_obs(Mode.OCULUS))

    assert policy.step_calls == 0
    assert policy.reset_calls == 1
    # Empty Action — every field is None so the env publishes nothing.
    assert action.joint_positions is None
    assert action.tool_pose is None
    assert action.base_pose is None
    assert action.gripper_positions is None


def test_none_mode_treated_as_active():
    """Observations without orchestrator info still drive step (standalone runs)."""
    policy = _RecordingPolicy()
    action = policy(_obs(None))

    assert policy.step_calls == 1
    assert policy.reset_calls == 0
    assert action.joint_positions is not None


def test_inactive_resets_every_call_for_clean_reactivation():
    """While the policy is inactive, reset must run each tick so its internal
    references re-anchor whenever it next becomes active."""
    policy = _RecordingPolicy()
    for _ in range(3):
        policy(_obs(Mode.KEYBOARD))

    expected_calls = 3
    assert policy.reset_calls == expected_calls
    assert policy.step_calls == 0
