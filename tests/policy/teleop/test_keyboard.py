"""Tests for KeyboardTeleopPolicy.

The pynput keyboard listener is patched out so tests don't require a display
or accessibility permissions; we exercise ``_on_press`` directly and
construct realistic ``Observation``s to drive ``__call__``.
"""

from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest
from pynput import keyboard as pynput_keyboard

from humanoid.config import ROBOT_CONFIGS
from humanoid.policy.teleop.keyboard import (
    KeyboardTeleopPolicy,
    KeyboardTeleopPolicyConfig,
)
from humanoid.types.observation import Observation
from humanoid.types.robot import RobotState


class _FakeCharKey:
    """Stand-in for pynput key events that carry a ``.char`` attribute."""

    def __init__(self, char: str):
        self.char = char


def _observation_from_q(q: np.ndarray) -> Observation:
    return Observation(
        robot_state=RobotState(
            timestamp=0.0,
            joint_positions=q.copy(),
            joint_velocities=np.zeros_like(q),
            motor_temperatures=np.zeros_like(q),
        )
    )


@pytest.fixture
def panda_policy(monkeypatch) -> KeyboardTeleopPolicy:
    """Keyboard policy on the fixed-base Panda with the listener disabled."""
    policy = KeyboardTeleopPolicy(
        robot_config=ROBOT_CONFIGS["panda"],
        config=KeyboardTeleopPolicyConfig(verbose=False),
    )
    monkeypatch.setattr(policy, "start_listener", lambda: None)
    monkeypatch.setattr(policy, "stop_listener", lambda: None)
    return policy


@pytest.fixture
def mobile_policy(monkeypatch) -> KeyboardTeleopPolicy:
    """Keyboard policy on the mobile robot (has a base frame)."""
    policy = KeyboardTeleopPolicy(
        robot_config=ROBOT_CONFIGS["elrobot_mobile"],
        config=KeyboardTeleopPolicyConfig(verbose=False),
    )
    monkeypatch.setattr(policy, "start_listener", lambda: None)
    monkeypatch.setattr(policy, "stop_listener", lambda: None)
    return policy


class TestConstruction:
    def test_default_config_used_when_none(self):
        policy = KeyboardTeleopPolicy(robot_config=ROBOT_CONFIGS["panda"], config=None)
        assert isinstance(policy.config, KeyboardTeleopPolicyConfig)

    def test_gripper_step_derived_from_range(self, panda_policy):
        # gripper_step = gripper_range * gripper_step_pct
        expected = (panda_policy.gripper_max - panda_policy.gripper_min) * (
            panda_policy.config.gripper_step_pct
        )
        assert panda_policy.gripper_step == pytest.approx(expected)

    def test_no_gripper_config_zeroes_step_and_limits(self):
        cfg = replace(ROBOT_CONFIGS["panda"], gripper_joint_indices=None)
        policy = KeyboardTeleopPolicy(
            robot_config=cfg, config=KeyboardTeleopPolicyConfig(verbose=False)
        )
        assert policy.gripper_step == 0.0
        assert policy.gripper_min == 0.0
        assert policy.gripper_max == 0.0


class TestFirstCallInitialization:
    def test_first_call_initializes_pose_gripper_and_starts_listener(
        self, panda_policy, monkeypatch
    ):
        started: list[bool] = []
        monkeypatch.setattr(panda_policy, "start_listener", lambda: started.append(True))

        obs = _observation_from_q(panda_policy.robot_config.home_position)
        action = panda_policy(obs)

        assert started == [True], "start_listener must run on the first call"
        assert panda_policy.current_pose is not None
        assert panda_policy.gripper_positions is not None
        assert action.tool_pose is not None
        # Initial target should match FK of the home position.
        expected = panda_policy.robot.get_tool_pose(panda_policy.robot_config.home_position)
        np.testing.assert_allclose(action.tool_pose.translation, expected.translation)

    def test_panda_first_call_has_no_base_pose(self, panda_policy):
        action = panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        assert action.base_pose is None
        assert panda_policy.current_base_pose is None

    def test_mobile_first_call_initializes_base_pose(self, mobile_policy):
        action = mobile_policy(_observation_from_q(mobile_policy.robot_config.home_position))
        assert action.base_pose is not None
        assert mobile_policy.current_base_pose is not None

    def test_action_pose_is_a_copy(self, panda_policy):
        """Mutating action.tool_pose must not bleed into the policy's state."""
        action = panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        before = panda_policy.current_pose.translation.copy()
        action.tool_pose.translation[0] += 1.0
        np.testing.assert_allclose(panda_policy.current_pose.translation, before)


class TestTranslationKeys:
    @pytest.mark.parametrize(
        "key,axis,sign",
        [
            ("w", 1, +1),  # +Y
            ("s", 1, -1),  # -Y
            ("a", 0, -1),  # -X
            ("d", 0, +1),  # +X
            ("q", 2, -1),  # -Z
            ("e", 2, +1),  # +Z
        ],
    )
    def test_translation_key_moves_along_expected_axis(self, panda_policy, key, axis, sign):
        panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        before = panda_policy.current_pose.translation.copy()

        panda_policy._on_press(_FakeCharKey(key))

        delta = panda_policy.current_pose.translation - before
        expected = np.zeros(3)
        expected[axis] = sign * panda_policy.translation_step
        np.testing.assert_allclose(delta, expected, atol=1e-12)


class TestRotationKeys:
    @pytest.mark.parametrize("key", ["i", "k", "j", "l", "u", "o"])
    def test_rotation_key_rotates_current_pose(self, panda_policy, key):
        panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        before = panda_policy.current_pose.rotation.copy()

        panda_policy._on_press(_FakeCharKey(key))

        # Rotation must change and remain a valid rotation matrix.
        assert not np.allclose(panda_policy.current_pose.rotation, before)
        np.testing.assert_allclose(
            panda_policy.current_pose.rotation @ panda_policy.current_pose.rotation.T,
            np.eye(3),
            atol=1e-10,
        )


class TestGripperKeys:
    def test_close_clamps_at_lower_limit(self, panda_policy):
        panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        # Slam the gripper closed; many steps' worth so we cross any limit.
        for _ in range(1000):
            panda_policy._on_press(_FakeCharKey("["))
        assert panda_policy.gripper_positions[0] == pytest.approx(panda_policy.gripper_min)

    def test_open_clamps_at_upper_limit(self, panda_policy):
        panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        for _ in range(1000):
            panda_policy._on_press(_FakeCharKey("]"))
        assert panda_policy.gripper_positions[0] == pytest.approx(panda_policy.gripper_max)

    def test_one_close_moves_by_gripper_step(self, panda_policy):
        panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        before = panda_policy.gripper_positions[0]
        panda_policy._on_press(_FakeCharKey("["))
        # Either moved by gripper_step or clamped at the floor.
        expected = max(panda_policy.gripper_min, before - panda_policy.gripper_step)
        assert panda_policy.gripper_positions[0] == pytest.approx(expected)


class TestBaseKeys:
    def test_base_arrow_keys_only_active_when_base_pose_present(self, panda_policy):
        """Panda has no base frame, so arrow keys must not crash and have no effect."""
        panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        # No current_base_pose to mutate; should silently no-op.
        panda_policy._on_press(pynput_keyboard.Key.up)
        assert panda_policy.current_base_pose is None

    def test_up_arrow_moves_base_forward(self, mobile_policy):
        mobile_policy(_observation_from_q(mobile_policy.robot_config.home_position))
        before = mobile_policy.current_base_pose.translation.copy()

        mobile_policy._on_press(pynput_keyboard.Key.up)

        delta = mobile_policy.current_base_pose.translation - before
        np.testing.assert_allclose(
            delta, [mobile_policy.base_translation_step, 0.0, 0.0], atol=1e-12
        )

    def test_left_arrow_strafes_base_positive_y(self, mobile_policy):
        mobile_policy(_observation_from_q(mobile_policy.robot_config.home_position))
        before = mobile_policy.current_base_pose.translation.copy()

        mobile_policy._on_press(pynput_keyboard.Key.left)

        delta = mobile_policy.current_base_pose.translation - before
        np.testing.assert_allclose(
            delta, [0.0, mobile_policy.base_translation_step, 0.0], atol=1e-12
        )

    def test_comma_yaws_base_positive(self, mobile_policy):
        mobile_policy(_observation_from_q(mobile_policy.robot_config.home_position))
        before = mobile_policy.current_base_pose.rotation.copy()

        mobile_policy._on_press(_FakeCharKey(","))

        expected = before @ pin.utils.rotate("z", mobile_policy.base_rotation_step)
        np.testing.assert_allclose(mobile_policy.current_base_pose.rotation, expected, atol=1e-12)


class TestQuitAndReset:
    def test_x_key_sets_running_false(self, panda_policy):
        panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        panda_policy._on_press(_FakeCharKey("x"))
        assert panda_policy.running is False

    def test_escape_key_sets_running_false(self, panda_policy):
        panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        panda_policy._on_press(pynput_keyboard.Key.esc)
        assert panda_policy.running is False

    def test_reset_clears_state_and_sets_running(self, panda_policy):
        panda_policy(_observation_from_q(panda_policy.robot_config.home_position))
        panda_policy.running = False

        panda_policy.reset()

        assert panda_policy.current_pose is None
        assert panda_policy.gripper_positions is None
        assert panda_policy.current_base_pose is None
        assert panda_policy.running is True
