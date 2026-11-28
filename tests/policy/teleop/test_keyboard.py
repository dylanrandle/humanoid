"""Tests for KeyboardTeleopPolicy.

The pynput keyboard listener is patched out so tests don't require a display
or accessibility permissions; we exercise ``_on_press`` directly and
construct realistic ``Observation``s to drive ``__call__``.
"""

from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

pytest.importorskip(
    "pynput",
    reason="pynput not available on this platform",
    exc_type=ImportError,
)
from pynput import keyboard as pynput_keyboard

from humanoid.config import ROBOT_CONFIGS
from humanoid.config.teleop import get_keyboard_teleop_policy_config
from humanoid.policy.teleop.keyboard import (
    KeyboardTeleopPolicy,
    KeyboardTeleopPolicyConfig,
)
from humanoid.types.homing import HomingPreset
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotJointCommand, RobotState


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
            actuator_temperatures=np.zeros_like(q),
        )
    )


@pytest.fixture
def panda_policy(monkeypatch) -> KeyboardTeleopPolicy:
    """Keyboard policy on the fixed-base Panda with the listener disabled."""
    policy = KeyboardTeleopPolicy(
        robot_config=ROBOT_CONFIGS["panda"],
        config=replace(get_keyboard_teleop_policy_config(), verbose=False),
    )
    monkeypatch.setattr(policy, "start_listener", lambda: None)
    monkeypatch.setattr(policy, "stop_listener", lambda: None)
    return policy


@pytest.fixture
def mobile_policy(monkeypatch) -> KeyboardTeleopPolicy:
    """Keyboard policy on the mobile robot (has a base frame)."""
    policy = KeyboardTeleopPolicy(
        robot_config=ROBOT_CONFIGS["triskel"],
        config=replace(get_keyboard_teleop_policy_config(), verbose=False),
    )
    monkeypatch.setattr(policy, "start_listener", lambda: None)
    monkeypatch.setattr(policy, "stop_listener", lambda: None)
    return policy


class TestConstruction:
    def test_default_config_used_when_none(self):
        policy = KeyboardTeleopPolicy(robot_config=ROBOT_CONFIGS["panda"], config=None)
        assert isinstance(policy.config, KeyboardTeleopPolicyConfig)

    @pytest.mark.parametrize("tool_scale,base_scale", [(0.1, 1.0), (0.05, 1.0), (0.1, 0.5)])
    def test_steps_use_robot_velocity_limits(self, tool_scale, base_scale):
        robot_config = ROBOT_CONFIGS["triskel"]
        config = replace(
            get_keyboard_teleop_policy_config(),
            tool_step_scale=tool_scale,
            base_step_scale=base_scale,
        )
        policy = KeyboardTeleopPolicy(robot_config=robot_config, config=config)
        step_seconds = config.dt * tool_scale

        assert robot_config.base is not None
        assert policy.tool_translation_step == pytest.approx(
            robot_config.tool.velocity_limits.linear * step_seconds
        )
        assert policy.tool_rotation_step == pytest.approx(
            robot_config.tool.velocity_limits.angular * step_seconds
        )
        assert policy.base_translation_step == pytest.approx(
            robot_config.base.velocity_limits.linear * config.dt * base_scale
        )
        assert policy.base_rotation_step == pytest.approx(
            robot_config.base.velocity_limits.angular * config.dt * base_scale
        )
        assert policy.gripper_step == pytest.approx(
            (policy.gripper_max - policy.gripper_min) * step_seconds / config.gripper_close_time
        )

    def test_no_gripper_config_zeroes_step_and_limits(self):
        cfg = replace(ROBOT_CONFIGS["panda"], gripper=None)
        policy = KeyboardTeleopPolicy(
            robot_config=cfg,
            config=replace(get_keyboard_teleop_policy_config(), verbose=False),
        )
        assert policy.gripper_step == 0.0
        assert policy.gripper_min == 0.0
        assert policy.gripper_max == 0.0


class TestFirstCallInitialization:
    def test_mode_entry_holds_commanded_arm_and_measured_base_until_first_jog(self, mobile_policy):
        policy = mobile_policy
        measured = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        measured[:4] = [0.2, -0.1, np.cos(0.3), np.sin(0.3)]
        previous = policy.robot_config.homing_presets[HomingPreset.REST].copy()
        gripper = policy.robot.get_gripper_position_indices()
        previous[gripper] = policy.gripper_max
        obs = _observation_from_q(measured)
        obs.robot_joint_command = RobotJointCommand(1.0, previous)
        obs.mode = Mode.QUEST
        assert policy(obs).base_pose is None

        obs.mode = Mode.KEYBOARD
        base = policy.robot.get_base_pose(measured).copy()
        tool = policy.robot.get_tool_command_pose(previous).copy()
        for _ in range(2):
            action = policy(obs)
            np.testing.assert_allclose(action.base_pose.homogeneous, base.homogeneous, atol=1e-12)
            np.testing.assert_allclose(action.tool_pose.homogeneous, tool.homogeneous, atol=1e-12)
            np.testing.assert_array_equal(action.gripper_positions, previous[gripper])
            obs.robot_state.joint_positions = previous.copy()

        policy._on_press(pynput_keyboard.Key.up)
        expected = base.translation + np.array([policy.base_translation_step, 0, 0])
        np.testing.assert_allclose(policy(obs).base_pose.translation, expected, atol=1e-12)

    @pytest.mark.parametrize("fixture", ["panda_policy", "mobile_policy"])
    def test_idle_holds_jog_target_across_feedback_updates(self, fixture, request):
        policy = request.getfixturevalue(fixture)
        home = policy.robot_config.homing_presets[HomingPreset.HOME]
        rest = policy.robot_config.homing_presets[HomingPreset.REST]
        obs = _observation_from_q(home)
        policy(obs)
        for key in ("d", "j", "["):
            policy._on_press(_FakeCharKey(key))
        policy._on_press(pynput_keyboard.Key.up)
        tool = policy.current_tool_pose.copy()
        base = policy.current_base_pose.copy() if policy.current_base_pose is not None else None
        gripper = policy.gripper_positions.copy()
        for _ in range(30):
            policy(obs)  # Allow the rate-limited jog to reach its target.

        policy.robot.forward_kinematics(rest)
        idle = policy(_observation_from_q(rest))
        np.testing.assert_allclose(idle.tool_pose.homogeneous, tool.homogeneous, atol=1e-12)
        np.testing.assert_allclose(idle.tool_velocity.linear, 0, atol=1e-12)
        np.testing.assert_allclose(idle.tool_velocity.angular, 0, atol=1e-12)
        np.testing.assert_array_equal(idle.gripper_positions, gripper)
        if base is not None:
            np.testing.assert_allclose(idle.base_pose.homogeneous, base.homogeneous, atol=1e-12)

    def test_first_call_initializes_pose_gripper_and_starts_listener(
        self, panda_policy, monkeypatch
    ):
        started: list[bool] = []
        monkeypatch.setattr(panda_policy, "start_listener", lambda: started.append(True))

        obs = _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        action = panda_policy(obs)

        assert started == [True], "start_listener must run on the first call"
        assert panda_policy.current_tool_pose is not None
        assert panda_policy.gripper_positions is not None
        assert action.tool_pose is not None
        # Initial target should match FK of the home position.
        expected = panda_policy.robot.get_tool_pose(
            panda_policy.robot_config.homing_presets[HomingPreset.HOME]
        )
        np.testing.assert_allclose(action.tool_pose.translation, expected.translation)

    def test_panda_first_call_has_no_base_pose(self, panda_policy):
        action = panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        assert action.base_pose is None
        assert panda_policy.current_base_pose is None

    def test_mobile_first_call_initializes_base_pose(self, mobile_policy):
        action = mobile_policy(
            _observation_from_q(mobile_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        assert action.base_pose is not None
        assert mobile_policy.current_base_pose is not None

    def test_action_pose_is_a_copy(self, panda_policy):
        """Mutating action.tool_pose must not bleed into the policy's state."""
        action = panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        before = panda_policy.current_tool_pose.translation.copy()
        action.tool_pose.translation[0] += 1.0
        np.testing.assert_allclose(panda_policy.current_tool_pose.translation, before)


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
        panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        before = panda_policy.current_tool_pose.translation.copy()

        panda_policy._on_press(_FakeCharKey(key))

        delta = panda_policy.current_tool_pose.translation - before
        expected = np.zeros(3)
        expected[axis] = sign * panda_policy.tool_translation_step
        np.testing.assert_allclose(delta, expected, atol=1e-12)


class TestRotationKeys:
    @pytest.mark.parametrize("key", ["i", "k", "j", "l", "u", "o"])
    def test_rotation_key_rotates_current_tool_pose(self, panda_policy, key):
        panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        before = panda_policy.current_tool_pose.rotation.copy()

        panda_policy._on_press(_FakeCharKey(key))

        # Rotation must change and remain a valid rotation matrix.
        assert not np.allclose(panda_policy.current_tool_pose.rotation, before)
        np.testing.assert_allclose(
            panda_policy.current_tool_pose.rotation @ panda_policy.current_tool_pose.rotation.T,
            np.eye(3),
            atol=1e-10,
        )


class TestGripperKeys:
    def test_triskel_close_and_open_from_home(self, mobile_policy):
        obs = _observation_from_q(mobile_policy.robot_config.homing_presets[HomingPreset.HOME])
        mobile_policy(obs)
        mobile_policy._on_press(_FakeCharKey("["))
        assert mobile_policy(obs).gripper_positions[0] == pytest.approx(mobile_policy.gripper_step)
        mobile_policy._on_press(_FakeCharKey("]"))
        assert mobile_policy(obs).gripper_positions[0] == pytest.approx(0)

    def test_close_clamps_at_lower_limit(self, panda_policy):
        panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        # Slam the gripper closed; many steps' worth so we cross any limit.
        for _ in range(1000):
            panda_policy._on_press(_FakeCharKey("["))
        assert panda_policy.gripper_positions[0] == pytest.approx(panda_policy.gripper_min)

    def test_open_clamps_at_upper_limit(self, panda_policy):
        panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        for _ in range(1000):
            panda_policy._on_press(_FakeCharKey("]"))
        assert panda_policy.gripper_positions[0] == pytest.approx(panda_policy.gripper_max)

    def test_one_close_moves_by_gripper_step(self, panda_policy):
        panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        before = panda_policy.gripper_positions[0]
        panda_policy._on_press(_FakeCharKey("["))
        # Either moved by gripper_step or clamped at the floor.
        expected = max(panda_policy.gripper_min, before - panda_policy.gripper_step)
        assert panda_policy.gripper_positions[0] == pytest.approx(expected)


class TestBaseKeys:
    def test_base_arrow_keys_only_active_when_base_pose_present(self, panda_policy):
        """Panda has no base frame, so arrow keys must not crash and have no effect."""
        panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        # No current_base_pose to mutate; should silently no-op.
        panda_policy._on_press(pynput_keyboard.Key.up)
        assert panda_policy.current_base_pose is None

    def test_up_arrow_moves_base_forward(self, mobile_policy):
        mobile_policy(
            _observation_from_q(mobile_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        before = mobile_policy.current_base_pose.translation.copy()

        mobile_policy._on_press(pynput_keyboard.Key.up)

        delta = mobile_policy.current_base_pose.translation - before
        np.testing.assert_allclose(
            delta, [mobile_policy.base_translation_step, 0.0, 0.0], atol=1e-12
        )

    def test_left_arrow_strafes_base_positive_y(self, mobile_policy):
        mobile_policy(
            _observation_from_q(mobile_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        before = mobile_policy.current_base_pose.translation.copy()

        mobile_policy._on_press(pynput_keyboard.Key.left)

        delta = mobile_policy.current_base_pose.translation - before
        np.testing.assert_allclose(
            delta, [0.0, mobile_policy.base_translation_step, 0.0], atol=1e-12
        )

    def test_comma_yaws_base_positive(self, mobile_policy):
        mobile_policy(
            _observation_from_q(mobile_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        before = mobile_policy.current_base_pose.rotation.copy()

        mobile_policy._on_press(_FakeCharKey(","))

        expected = before @ pin.utils.rotate("z", mobile_policy.base_rotation_step)
        np.testing.assert_allclose(mobile_policy.current_base_pose.rotation, expected, atol=1e-12)


class TestQuitAndReset:
    def test_x_key_sets_running_false(self, panda_policy):
        panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        panda_policy._on_press(_FakeCharKey("x"))
        assert panda_policy.running is False

    def test_escape_key_sets_running_false(self, panda_policy):
        panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        panda_policy._on_press(pynput_keyboard.Key.esc)
        assert panda_policy.running is False

    def test_reset_clears_state_and_sets_running(self, panda_policy):
        panda_policy(
            _observation_from_q(panda_policy.robot_config.homing_presets[HomingPreset.HOME])
        )
        panda_policy.running = False

        panda_policy.reset()

        assert panda_policy.current_tool_pose is None
        assert panda_policy.gripper_positions is None
        assert panda_policy.current_base_pose is None
        assert panda_policy.running is True
