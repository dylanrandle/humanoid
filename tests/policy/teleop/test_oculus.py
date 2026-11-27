"""Tests for OculusTeleopPolicy.

The ``oculus_reader.OculusReader`` device dependency is patched out with a
mutable stub so we can drive the policy through synthetic controller poses,
button states, and joystick deflections.
"""

import time
import typing
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.config.teleop import TELEOP_SUBSCRIPTIONS
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.policy.teleop import oculus as oculus_module
from humanoid.policy.teleop.oculus import (
    OculusTeleopPolicy,
    OculusTeleopPolicyConfig,
)
from humanoid.types.action import Action
from humanoid.types.homing import HomingPreset
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.recording import RecordingState, RecordingStatus
from humanoid.types.robot import RobotJointCommand, RobotState
from humanoid.types.teleop import OculusInputSnapshot

ONE_CALL = 1
TWO_CALLS = 2
CUSTOM_ADB_PORT = 5556


class StubOculusReader:
    """Stand-in for oculus_reader.OculusReader.

    Returns whatever pose/button state the test has loaded into it. Tests
    mutate ``transforms`` and ``buttons`` between policy calls to simulate
    operator input.
    """

    def __init__(self, ip_address: str | None = None, port: int = 5555):
        self.ip_address = ip_address
        self.port = port
        self.input_fresh = True
        self.captured_monotonic: float | None = None
        self.last_disengaged_monotonic: float | None = None
        self.transforms: dict = {"l": np.eye(4), "r": np.eye(4)}
        self.buttons: dict = {
            "A": False,
            "B": False,
            "X": False,
            "Y": False,
            "LG": False,
            "RG": False,
            "leftJS": (0.0, 0.0),
            "rightJS": (0.0, 0.0),
        }

    def get_snapshot(self):
        captured_monotonic = time.monotonic() if self.input_fresh else self.captured_monotonic
        return OculusInputSnapshot(
            transforms=self.transforms,
            buttons=self.buttons,
            captured_monotonic=captured_monotonic,
            last_disengaged_monotonic=self.last_disengaged_monotonic,
        )


class StubOrchestratorClient(OrchestratorClient):
    """Stand-in for OrchestratorClient that records ``request_homing`` calls."""

    start_recording: typing.Any
    stop_recording: typing.Any

    def __init__(self):
        self.homing_calls: list[np.ndarray] = []
        self.start_recording = lambda: None
        self.stop_recording = lambda: None

    def request_homing(self, target_position: np.ndarray) -> None:
        self.homing_calls.append(np.asarray(target_position).copy())


@pytest.fixture(autouse=True)
def _patch_reader(monkeypatch):
    """Auto-applied: every test in this module sees the stub, not real hardware."""
    monkeypatch.setattr(oculus_module, "OculusReader", StubOculusReader)


def _observation_from_q(q: np.ndarray) -> Observation:
    return Observation(
        robot_state=RobotState(
            timestamp=0.0,
            joint_positions=q.copy(),
            joint_velocities=np.zeros_like(q),
            actuator_temperatures=np.zeros_like(q),
        )
    )


def _make_policy(
    *,
    robot_name: str = "panda",
    config: OculusTeleopPolicyConfig | None = None,
) -> tuple[OculusTeleopPolicy, StubOculusReader]:
    robot_config = ROBOT_CONFIGS[robot_name]
    if config is None:
        config = OculusTeleopPolicyConfig(
            subscriptions=TELEOP_SUBSCRIPTIONS,
            verbose=False,
            oculus_to_tool_command_rotation=np.eye(3),
        )
    policy = OculusTeleopPolicy(
        robot_config=robot_config,
        config=config,
        orchestrator_client=StubOrchestratorClient(),
    )
    return policy, policy.reader  # ty:ignore[invalid-return-type]


@pytest.fixture
def panda_policy_and_reader():
    return _make_policy(robot_name="panda")


@pytest.fixture
def mobile_policy_and_reader():
    return _make_policy(robot_name="triskel")


class TestConstruction:
    def test_uses_stub_reader(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        assert isinstance(reader, StubOculusReader)
        assert policy.config.verbose is False

    def test_default_mapping_rotates_oculus_y_and_z_axes(self):
        config = OculusTeleopPolicyConfig(
            subscriptions=TELEOP_SUBSCRIPTIONS,
        )

        np.testing.assert_array_equal(
            config.oculus_to_tool_command_rotation,
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        )

    def test_passes_wireless_connection_to_reader(self):
        config = OculusTeleopPolicyConfig(
            subscriptions=TELEOP_SUBSCRIPTIONS,
            ip_address="192.168.1.42",
            port=CUSTOM_ADB_PORT,
            verbose=False,
        )

        policy, reader = _make_policy(config=config)

        assert policy.config.ip_address == "192.168.1.42"
        assert reader.ip_address == "192.168.1.42"
        assert reader.port == CUSTOM_ADB_PORT

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"ip_address": "not-an-ip"}, "valid IPv4"),
            ({"port": 0}, "port"),
            ({"input_timeout": 0.0}, "input timeout"),
            ({"startup_timeout": float("inf")}, "startup timeout"),
        ],
    )
    def test_rejects_invalid_connection_config(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            OculusTeleopPolicyConfig(subscriptions=TELEOP_SUBSCRIPTIONS, **kwargs)

    def test_mobile_base_limits_come_from_robot_config(self):
        robot_config = ROBOT_CONFIGS["triskel"]

        policy = OculusTeleopPolicy(
            robot_config=robot_config,
            config=None,
            orchestrator_client=StubOrchestratorClient(),
        )

        assert robot_config.base is not None
        assert policy.base_config is robot_config.base


class TestInputDropout:
    @pytest.mark.parametrize("missing", ["l", "r", "LG", "RG", "both"])
    def test_missing_controller_suppresses_all_commands(self, mobile_policy_and_reader, missing):
        policy, reader = mobile_policy_and_reader
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        reader.buttons.update(RG=True, LG=True, leftJS=(1.0, 0.0), A=True)
        assert policy(obs).tool_pose is not None
        if missing == "both":
            reader.transforms.clear()
        elif missing in reader.transforms:
            del reader.transforms[missing]
        else:
            del reader.buttons[missing]

        client = StubOrchestratorClient()
        client.start_recording = MagicMock()
        policy.orchestrator_client = client
        reader.buttons.update(X=True, Y=True)
        for _ in range(2):
            assert policy(obs) == Action()
        assert not client.homing_calls
        client.start_recording.assert_not_called()
        assert policy.filtered_controller_pose is None

    def test_homing_return_with_missing_left_hand_does_not_repeat_held_button(
        self, mobile_policy_and_reader
    ):
        policy, reader = mobile_policy_and_reader
        client = StubOrchestratorClient()
        policy.orchestrator_client = client
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        obs.mode = Mode.OCULUS
        reader.buttons.update(X=True, LG=True)
        policy(obs)
        obs.mode = Mode.HOMING
        assert policy(obs).tool_pose is None

        obs.mode = Mode.OCULUS
        left_pose = reader.transforms.pop("l")
        assert policy(obs) == Action()
        reader.transforms["r"][0, 3] += 0.15
        reader.transforms["l"] = left_pose
        reader.buttons["RG"] = True
        _assert_se3_equal(
            policy(obs).tool_pose,
            policy.robot.get_tool_command_pose(obs.robot_state.joint_positions),
        )
        assert len(client.homing_calls) == ONE_CALL

        reader.buttons["X"] = False
        policy(obs)
        reader.buttons["X"] = True
        policy(obs)
        assert len(client.homing_calls) == TWO_CALLS

    def test_stale_frame_suppresses_commands_and_clears_motion_state(
        self, mobile_policy_and_reader, caplog
    ):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        reader.buttons["leftJS"] = (1.0, 0.0)
        observation = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        policy(observation)
        assert policy.filtered_controller_pose is not None

        reader.input_fresh = False
        reader.captured_monotonic = time.monotonic() - policy.config.input_timeout - 0.1
        action = policy(observation)

        assert "controller stream timed out" in caplog.text
        assert policy.filtered_controller_pose is None
        assert action == Action()

    def test_fresh_input_after_stale_frame_reanchors(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.input_fresh = False
        reader.captured_monotonic = time.monotonic() - policy.config.input_timeout - 0.1
        observation = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        policy(observation)

        reader.input_fresh = True
        reader.buttons["RG"] = True
        action = policy(observation)

        assert policy.filtered_controller_pose is not None
        _assert_se3_equal(
            action.tool_pose,
            policy.robot.get_tool_command_pose(observation.robot_state.joint_positions),
        )

    def test_recovery_reanchors_all_motion_state(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        reader.buttons["A"] = True
        reader.buttons["leftJS"] = (1.0, 0.0)

        initial_q = policy.robot_config.homing_presets[HomingPreset.HOME]
        policy(_observation_from_q(initial_q))

        reader.transforms["r"] = _se3_to_matrix(pin.SE3(np.eye(3), np.array([0.004, 0.0, 0.0])))
        policy(_observation_from_q(initial_q))
        assert policy.filtered_controller_pose is not None
        assert policy.commanded_tool_pose is not None
        assert policy.reference_base_pose is not None
        assert policy.commanded_gripper_position is not None

        recovery_q = policy.robot_config.homing_presets[HomingPreset.REST].copy()
        recovery_yaw = 0.4
        recovery_q[:4] = [1.0, -0.5, np.cos(recovery_yaw), np.sin(recovery_yaw)]
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        recovery_gripper = (policy.gripper_min + policy.gripper_max) / 2
        recovery_q[gripper_position_idx] = recovery_gripper
        recovery_observation = _observation_from_q(recovery_q)

        reader.transforms = {}
        reader.buttons = {}
        hold_action = policy(recovery_observation)

        assert policy.filtered_controller_pose is None
        assert policy.commanded_tool_pose is None
        assert policy.reference_base_pose is None
        assert policy.commanded_gripper_position is None
        assert hold_action == Action()

        recovered_reader_state = StubOculusReader()
        reader.transforms = recovered_reader_state.transforms
        reader.buttons = recovered_reader_state.buttons
        reader.buttons["RG"] = True
        recovered_action = policy(recovery_observation)

        _assert_se3_equal(
            recovered_action.tool_pose,
            policy.robot.get_tool_command_pose(recovery_q),
        )
        _assert_se3_equal(recovered_action.base_pose, policy.robot.get_base_pose(recovery_q))
        assert recovered_action.gripper_positions is not None
        assert recovered_action.gripper_positions[0] == pytest.approx(recovery_gripper)


class TestDeadManSwitch:
    @pytest.mark.parametrize("engaged", [False, True])
    def test_mode_entry_holds_commanded_arm_and_measured_base(
        self, mobile_policy_and_reader, engaged
    ):
        policy, reader = mobile_policy_and_reader
        measured = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        measured[:4] = [0.2, -0.1, np.cos(0.3), np.sin(0.3)]
        previous = policy.robot_config.homing_presets[HomingPreset.REST].copy()
        obs = _observation_from_q(measured)
        obs.robot_joint_command = RobotJointCommand(1.0, previous)
        obs.mode = Mode.KEYBOARD
        assert policy(obs).base_pose is None
        obs.mode = Mode.OCULUS
        expected_base = policy.robot.get_base_pose(measured).copy()
        expected_tool = policy.robot.get_tool_command_pose(previous).copy()
        gripper = policy.robot.get_gripper_position_indices()
        for grip in (engaged, False, True):
            reader.buttons["RG"] = grip
            action = policy(obs)
            _assert_se3_equal(action.base_pose, expected_base)
            _assert_se3_equal(action.tool_pose, expected_tool)
            np.testing.assert_array_equal(action.gripper_positions, previous[gripper])

        obs.robot_joint_command = RobotJointCommand(2.0, previous)
        reader.buttons["RG"] = False
        _assert_se3_equal(policy(obs).base_pose, policy.robot.get_base_pose(previous))

    def test_release_latches_targets_until_next_grip(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        home = policy.robot_config.homing_presets[HomingPreset.HOME]
        obs = _observation_from_q(home)
        reader.buttons["RG"] = True
        policy(obs)

        release_q = policy.robot_config.homing_presets[HomingPreset.REST].copy()
        release_q[:4] = [0.2, -0.1, 0, 1]
        obs.robot_joint_command = RobotJointCommand(0, release_q.copy())
        reader.buttons["RG"] = False
        policy(obs)
        expected_tool = policy.robot.get_tool_command_pose(release_q).copy()
        expected_base = policy.robot.get_base_pose(release_q).copy()
        expected_gripper = release_q[policy.robot.get_gripper_position_indices()]

        # Incoming joint commands and subsequent FK calls must not move the hold.
        obs.robot_joint_command.joint_positions[:] = home
        policy.robot.forward_kinematics(home)
        for _ in range(3):
            held = policy(obs)
            _assert_se3_equal(held.tool_pose, expected_tool)
            _assert_se3_equal(held.base_pose, expected_base)
            np.testing.assert_array_equal(held.gripper_positions, expected_gripper)
            assert held.tool_velocity is None

        reader.buttons["LG"] = True
        _assert_se3_equal(policy(obs).tool_pose, policy.robot.get_tool_command_pose(home))
        reader.buttons["LG"] = False
        _assert_se3_equal(policy(obs).tool_pose, policy.robot.get_tool_command_pose(home))

    def test_grip_released_returns_hold_action_and_clears_references(self, panda_policy_and_reader):
        policy, _ = panda_policy_and_reader
        # Pre-populate references to make sure release wipes them.
        policy.filtered_controller_pose = pin.SE3.Identity()
        policy.commanded_tool_pose = pin.SE3.Identity()

        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        action = policy(obs)

        assert policy.filtered_controller_pose is None
        assert policy.commanded_tool_pose is None

        expected_tool = policy.robot.get_tool_pose(
            policy.robot_config.homing_presets[HomingPreset.HOME]
        )
        np.testing.assert_allclose(action.tool_pose.translation, expected_tool.translation)

    @pytest.mark.parametrize(
        "grip_buttons",
        [{"LG": True, "RG": False}, {"LG": False, "RG": True}, {"LG": True, "RG": True}],
    )
    def test_either_grip_engages_motion(self, panda_policy_and_reader, grip_buttons):
        policy, reader = panda_policy_and_reader
        reader.buttons.update(grip_buttons)
        # The command-frame remap is identity here, so the controller-frame
        # delta is unchanged.
        reader.transforms["r"] = _se3_to_matrix(pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.0])))

        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        action = policy(obs)

        # References must be set after a single engaged tick.
        assert policy.filtered_controller_pose is not None
        assert policy.commanded_tool_pose is not None
        assert action.tool_pose is not None


class TestControllerDeltaApplied:
    def test_release_between_policy_ticks_reanchors_at_new_hand_pose(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        initial = policy(obs)
        reader.last_disengaged_monotonic = time.monotonic()
        reader.transforms["r"][0, 3] += 0.3
        for _ in range(3):
            _assert_se3_equal(policy(obs).tool_pose, initial.tool_pose)

    def test_large_hand_step_does_not_queue_motion_while_stationary(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        policy(obs)
        reader.transforms["r"] = _se3_to_matrix(
            pin.SE3(pin.utils.rotate("z", np.pi / 2), np.array([0.5, 0.0, 0.0]))
        )
        moved = policy(obs)
        for _ in range(10):
            _assert_se3_equal(policy(obs).tool_pose, moved.tool_pose)

    def test_no_motion_yields_reference_tool_pose(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        # Initial controller pose at identity.
        reader.transforms["r"] = np.eye(4)

        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        action = policy(obs)

        expected = policy.robot.get_tool_pose(policy.robot_config.homing_presets[HomingPreset.HOME])
        np.testing.assert_allclose(action.tool_pose.translation, expected.translation, atol=1e-12)

    def test_controller_translation_translates_target(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        # First tick captures the reference (identity).
        ref = policy(
            _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        ).tool_pose

        # Keep the delta below the robot's per-tick tool velocity limit.
        delta = np.array([0.001, 0.0, 0.0])
        reader.transforms["r"] = _se3_to_matrix(pin.SE3(np.eye(3), delta))

        action = policy(_observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME]))

        expected_translation = ref.translation + delta
        np.testing.assert_allclose(action.tool_pose.translation, expected_translation, atol=1e-12)

    def test_translation_scale_amplifies_delta(self):
        policy, reader = _make_policy(
            config=OculusTeleopPolicyConfig(
                subscriptions=TELEOP_SUBSCRIPTIONS,
                verbose=False,
                tool_translation_scale=3.0,
                oculus_to_tool_command_rotation=np.eye(3),
            ),
        )
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        ref = policy(obs).tool_pose  # capture reference

        delta = np.array([0.001, 0.0, 0.0])
        reader.transforms["r"] = _se3_to_matrix(pin.SE3(np.eye(3), delta))
        action = policy(obs)

        expected_translation = ref.translation + delta * 3.0  # ty:ignore[unresolved-attribute]
        np.testing.assert_allclose(action.tool_pose.translation, expected_translation, atol=1e-12)  # ty:ignore[unresolved-attribute]

    def test_oculus_to_tool_command_remap_applied_to_delta(self):
        """A 90° rotation about Z should swap x and y axes of the input delta."""
        rotation_about_z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        policy, reader = _make_policy(
            config=OculusTeleopPolicyConfig(
                subscriptions=TELEOP_SUBSCRIPTIONS,
                verbose=False,
                oculus_to_tool_command_rotation=rotation_about_z_90,
            ),
        )
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        ref = policy(obs).tool_pose

        # +x in the Oculus frame should appear as +y after the remap.
        reader.transforms["r"] = _se3_to_matrix(pin.SE3(np.eye(3), np.array([0.001, 0.0, 0.0])))
        action = policy(obs)

        remapped_delta = rotation_about_z_90 @ np.array([0.001, 0.0, 0.0])
        expected_translation = ref.translation + remapped_delta  # ty:ignore[unresolved-attribute]
        np.testing.assert_allclose(action.tool_pose.translation, expected_translation, atol=1e-12)  # ty:ignore[unresolved-attribute]

    @pytest.mark.parametrize(
        ("oculus_delta", "command_delta"),
        [
            (np.array([0.0, 0.001, 0.0]), np.array([0.0, 0.0, 0.001])),
            (np.array([0.0, 0.0, 0.001]), np.array([0.0, -0.001, 0.0])),
        ],
    )
    def test_default_mapping_rotates_up_and_forward_axes(self, oculus_delta, command_delta):
        policy, _ = _make_policy(
            config=OculusTeleopPolicyConfig(subscriptions=TELEOP_SUBSCRIPTIONS, verbose=False)
        )
        policy.filtered_controller_pose = pin.SE3.Identity()
        policy.commanded_tool_pose = pin.SE3.Identity()

        target = policy._compute_tool_pose(_se3_to_matrix(pin.SE3(np.eye(3), oculus_delta)))

        np.testing.assert_allclose(target.translation, command_delta, atol=1e-12)

    @pytest.mark.parametrize("robot_name", ["panda", "triskel"])
    def test_translation_uses_command_frame_with_rotated_references(self, robot_name):
        policy, _ = _make_policy(robot_name=robot_name)
        controller_reference = pin.SE3(
            pin.utils.rotate("z", np.pi / 2),
            np.array([0.2, -0.1, 0.3]),
        )
        tool_reference = pin.SE3(
            pin.utils.rotate("y", np.pi / 2),
            np.array([0.4, 0.2, 0.5]),
        )
        tracking_delta = np.array([0.001, 0.0, 0.0])
        current_controller = pin.SE3(
            controller_reference.rotation,
            controller_reference.translation + tracking_delta,
        )
        policy.filtered_controller_pose = controller_reference
        policy.commanded_tool_pose = tool_reference.copy()

        target = policy._compute_tool_pose(_se3_to_matrix(current_controller))

        np.testing.assert_allclose(
            target.translation,
            tool_reference.translation + tracking_delta,
            atol=1e-12,
        )

    def test_rotation_uses_command_frame_with_rotated_references(self, panda_policy_and_reader):
        policy, _ = panda_policy_and_reader
        controller_reference = pin.SE3(
            pin.utils.rotate("y", np.pi / 2),
            np.array([0.2, -0.1, 0.3]),
        )
        tool_reference = pin.SE3(
            pin.utils.rotate("x", np.pi / 2),
            np.array([0.4, 0.2, 0.5]),
        )
        tracking_rotation_delta = pin.utils.rotate("z", 0.01)
        current_controller = pin.SE3(
            tracking_rotation_delta @ controller_reference.rotation,
            controller_reference.translation,
        )
        policy.filtered_controller_pose = controller_reference
        policy.commanded_tool_pose = tool_reference.copy()

        target = policy._compute_tool_pose(_se3_to_matrix(current_controller))

        np.testing.assert_allclose(
            target.rotation,
            tracking_rotation_delta @ tool_reference.rotation,
            atol=1e-12,
        )

    def test_oculus_to_tool_command_remap_applied_to_rotation(self):
        command_rotation = pin.utils.rotate("z", np.pi / 2)
        policy, _ = _make_policy(
            config=OculusTeleopPolicyConfig(
                subscriptions=TELEOP_SUBSCRIPTIONS,
                verbose=False,
                oculus_to_tool_command_rotation=command_rotation,
            ),
        )
        tool_reference = pin.SE3.Identity()
        tracking_rotation_delta = pin.utils.rotate("x", 0.01)
        policy.filtered_controller_pose = pin.SE3.Identity()
        policy.commanded_tool_pose = tool_reference.copy()

        target = policy._compute_tool_pose(
            _se3_to_matrix(pin.SE3(tracking_rotation_delta, np.zeros(3)))
        )

        expected_rotation_delta = command_rotation @ tracking_rotation_delta @ command_rotation.T
        np.testing.assert_allclose(target.rotation, expected_rotation_delta, atol=1e-12)

    def test_reengagement_orientation_does_not_change_translation_direction(
        self, panda_policy_and_reader
    ):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        tracking_delta = np.array([0.001, 0.0, 0.0])

        first_reference = pin.SE3.Identity()
        reader.transforms["r"] = _se3_to_matrix(first_reference)
        first_anchor = policy(obs).tool_pose
        reader.transforms["r"] = _se3_to_matrix(
            pin.SE3(first_reference.rotation, first_reference.translation + tracking_delta)
        )
        first_target = policy(obs).tool_pose

        reader.buttons["RG"] = False
        policy(obs)

        second_reference = pin.SE3(pin.utils.rotate("z", np.pi / 2), np.zeros(3))
        reader.buttons["RG"] = True
        reader.transforms["r"] = _se3_to_matrix(second_reference)
        second_anchor = policy(obs).tool_pose
        reader.transforms["r"] = _se3_to_matrix(
            pin.SE3(second_reference.rotation, second_reference.translation + tracking_delta)
        )
        second_target = policy(obs).tool_pose

        assert first_anchor is not None
        assert first_target is not None
        assert second_anchor is not None
        assert second_target is not None
        np.testing.assert_allclose(
            first_target.translation - first_anchor.translation,
            tracking_delta,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            second_target.translation - second_anchor.translation,
            tracking_delta,
            atol=1e-12,
        )

    def test_translation_is_limited_by_robot_config(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        ref = policy(obs).tool_pose

        reader.transforms["r"] = _se3_to_matrix(pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])))
        action = policy(obs)

        assert ref is not None
        translation_delta = action.tool_pose.translation - ref.translation
        max_translation = policy.robot_config.tool.velocity_limits.linear * policy.config.dt
        np.testing.assert_allclose(
            translation_delta,
            np.array([max_translation, 0.0, 0.0]),
            atol=1e-12,
        )

    def test_rotation_is_limited_by_robot_config(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        ref = policy(obs).tool_pose

        reader.transforms["r"] = _se3_to_matrix(
            pin.SE3(pin.utils.rotate("z", np.pi / 2), np.zeros(3))
        )
        action = policy(obs)

        assert ref is not None
        rotation_delta = action.tool_pose.rotation @ ref.rotation.T
        rotation_vector = pin.log3(rotation_delta)
        max_rotation = policy.robot_config.tool.velocity_limits.angular * policy.config.dt
        np.testing.assert_allclose(
            rotation_vector,
            np.array([0.0, 0.0, max_rotation]),
            atol=1e-12,
        )


class TestGripper:
    def test_first_tick_seeds_from_observation(self, panda_policy_and_reader):
        """With neither A nor B held, the commanded gripper equals the observation."""
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True

        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        seeded_gripper = 0.0123
        q[gripper_position_idx] = seeded_gripper

        action = policy(_observation_from_q(q))

        assert action.gripper_positions is not None
        assert action.gripper_positions[0] == pytest.approx(seeded_gripper)

    def test_a_button_steps_gripper_closed_by_one_step(self, panda_policy_and_reader):
        """A closes the gripper by moving toward its lower joint limit."""
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True

        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        seeded_gripper = 0.02
        q[gripper_position_idx] = seeded_gripper
        obs = _observation_from_q(q)

        reader.buttons["A"] = True
        action = policy(obs)

        step = policy.gripper_step
        expected = seeded_gripper - step
        assert action.gripper_positions[0] == pytest.approx(expected)

    def test_b_button_steps_gripper_open_by_one_step(self, panda_policy_and_reader):
        """B opens the gripper by moving toward its upper joint limit."""
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True

        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        seeded_gripper = 0.02
        q[gripper_position_idx] = seeded_gripper
        obs = _observation_from_q(q)

        reader.buttons["B"] = True
        action = policy(obs)

        step = policy.gripper_step
        expected = seeded_gripper + step
        assert action.gripper_positions[0] == pytest.approx(expected)

    def test_triskel_close_and_open_from_home(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        reader.buttons.update(RG=True, A=True)
        assert policy(obs).gripper_positions[0] == pytest.approx(policy.gripper_step)
        reader.buttons.update(A=False, B=True)
        assert policy(obs).gripper_positions[0] == pytest.approx(0)

    def test_repeated_a_holds_accumulate(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True

        # Seed at the upper limit so several closing steps don't hit the lower clamp.
        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        seeded_gripper = policy.gripper_max
        q[gripper_position_idx] = seeded_gripper
        obs = _observation_from_q(q)

        reader.buttons["A"] = True
        num_ticks = 3
        for _ in range(num_ticks):
            action = policy(obs)

        step = policy.gripper_step
        expected = seeded_gripper - num_ticks * step
        assert action.gripper_positions[0] == pytest.approx(expected)

    def test_release_holds_last_commanded_value(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True

        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        seeded_gripper = 0.02
        q[gripper_position_idx] = seeded_gripper
        obs = _observation_from_q(q)

        reader.buttons["A"] = True
        policy(obs)
        commanded_after_step = policy.commanded_gripper_position

        reader.buttons["A"] = False
        action = policy(obs)

        assert action.gripper_positions[0] == pytest.approx(commanded_after_step)

    def test_a_button_clamps_at_gripper_min(self, panda_policy_and_reader):
        """A closes by subtracting a step, so it clamps at the lower limit."""
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        # Seed near the lower limit so a single step would overshoot.
        step = policy.gripper_step
        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        q[policy.robot.get_gripper_position_indices()] = policy.gripper_min + 0.5 * step

        reader.buttons["A"] = True
        action = policy(_observation_from_q(q))

        assert action.gripper_positions[0] == pytest.approx(policy.gripper_min)

    def test_b_button_clamps_at_gripper_max(self, panda_policy_and_reader):
        """B opens by adding a step, so it clamps at the upper limit."""
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        step = policy.gripper_step
        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        q[policy.robot.get_gripper_position_indices()] = policy.gripper_max - 0.5 * step

        reader.buttons["B"] = True
        action = policy(_observation_from_q(q))

        assert action.gripper_positions[0] == pytest.approx(policy.gripper_max)

    def test_a_button_takes_priority_over_b(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True

        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        seeded_gripper = 0.02
        q[gripper_position_idx] = seeded_gripper
        obs = _observation_from_q(q)

        reader.buttons["A"] = True
        reader.buttons["B"] = True
        action = policy(obs)

        step = policy.gripper_step
        expected = seeded_gripper - step
        assert action.gripper_positions[0] == pytest.approx(expected)

    def test_gripper_buttons_ignored_without_deadman(self, panda_policy_and_reader):
        """A/B do nothing while no grip is held; gripper holds observation."""
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = False
        reader.buttons["LG"] = False

        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        seeded_gripper = 0.02
        q[gripper_position_idx] = seeded_gripper
        obs = _observation_from_q(q)

        reader.buttons["A"] = True
        action = policy(obs)

        # Without a grip trigger, the gripper stays at the observation value.
        assert action.gripper_positions[0] == pytest.approx(seeded_gripper)
        assert policy.commanded_gripper_position is None

    def test_deadman_release_clears_commanded_gripper(self, panda_policy_and_reader):
        """Releasing the dead-man re-anchors gripper to the observation on re-engage."""
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True

        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        seeded_gripper = 0.02
        q[gripper_position_idx] = seeded_gripper
        obs = _observation_from_q(q)

        reader.buttons["A"] = True
        policy(obs)
        assert policy.commanded_gripper_position is not None

        # Release dead-man -- commanded state should reset.
        reader.buttons["RG"] = False
        policy(obs)
        assert policy.commanded_gripper_position is None

        # Re-engaging with A still held seeds from observation, then steps once.
        reader.buttons["RG"] = True
        action = policy(obs)
        step = policy.gripper_step
        assert action.gripper_positions[0] == pytest.approx(seeded_gripper - step)

    def test_no_gripper_indices_yields_none(self):
        cfg = replace(ROBOT_CONFIGS["panda"], gripper=None)
        policy = OculusTeleopPolicy(
            robot_config=cfg,
            config=OculusTeleopPolicyConfig(
                subscriptions=TELEOP_SUBSCRIPTIONS,
                verbose=False,
                oculus_to_tool_command_rotation=np.eye(3),
            ),
            orchestrator_client=StubOrchestratorClient(),
        )
        policy.reader.buttons["RG"] = True  # ty:ignore[unresolved-attribute]
        policy.reader.buttons["A"] = True  # ty:ignore[unresolved-attribute]

        action = policy(_observation_from_q(cfg.homing_presets[HomingPreset.HOME]))

        assert action.gripper_positions is None


class TestBasePoseFromJoysticks:
    @pytest.mark.parametrize("axes", [(0.5,), (float("nan"), 0.0)])
    def test_invalid_joystick_sample_holds_base(self, mobile_policy_and_reader, axes):
        policy, reader = mobile_policy_and_reader
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        reader.buttons.update(RG=True, leftJS=axes, rightJS=axes)
        _assert_se3_equal(
            policy(obs).base_pose, policy.robot.get_base_pose(obs.robot_state.joint_positions)
        )

    def test_no_base_config_yields_no_base_pose(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        reader.buttons["leftJS"] = (1.0, 1.0)
        reader.buttons["rightJS"] = (1.0, 0.0)

        action = policy(_observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME]))

        assert action.base_pose is None

    def test_first_engaged_call_anchors_base_from_fk(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        # Zero stick → no translation/rotation delta.
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])

        action = policy(obs)

        expected = policy.robot.get_base_pose(policy.robot_config.homing_presets[HomingPreset.HOME])
        np.testing.assert_allclose(action.base_pose.translation, expected.translation, atol=1e-12)

    def test_left_stick_forward_drives_base_along_plus_y(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        policy(obs)  # initialize base anchor
        before = policy.reference_base_pose.translation.copy()

        reader.buttons["leftJS"] = (0.0, 1.0)
        policy(obs)

        assert policy.base_config is not None
        translation_step = policy.base_config.velocity_limits.linear * policy.config.dt
        delta = policy.reference_base_pose.translation - before
        np.testing.assert_allclose(
            delta,
            [0.0, translation_step, 0.0],
            atol=1e-12,
        )

    def test_left_stick_right_drives_base_along_plus_x(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        policy(obs)
        before = policy.reference_base_pose.translation.copy()

        reader.buttons["leftJS"] = (1.0, 0.0)
        policy(obs)

        assert policy.base_config is not None
        translation_step = policy.base_config.velocity_limits.linear * policy.config.dt
        delta = policy.reference_base_pose.translation - before
        np.testing.assert_allclose(
            delta,
            [translation_step, 0.0, 0.0],
            atol=1e-12,
        )

    def test_right_stick_right_yields_negative_yaw(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        policy(obs)
        before = policy.reference_base_pose.rotation.copy()

        reader.buttons["rightJS"] = (1.0, 0.0)
        policy(obs)

        # base_yaw_scale defaults to -1, so +jx -> negative yaw.
        assert policy.base_config is not None
        rotation_step = policy.base_config.velocity_limits.angular * policy.config.dt
        expected_dyaw = policy.config.base_yaw_scale * 1.0 * rotation_step
        expected = before @ pin.utils.rotate("z", expected_dyaw)
        np.testing.assert_allclose(policy.reference_base_pose.rotation, expected, atol=1e-12)

    def test_joystick_deadzone_suppresses_small_input(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        policy(obs)
        before = policy.reference_base_pose.translation.copy()

        # Well inside the default deadzone (0.1).
        reader.buttons["leftJS"] = (0.05, 0.05)
        policy(obs)

        np.testing.assert_allclose(policy.reference_base_pose.translation, before, atol=1e-12)


class TestHomingButtons:
    def test_x_button_requests_homing_to_home_position(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        stub_client = StubOrchestratorClient()
        policy.orchestrator_client = stub_client
        reader.buttons["X"] = True

        # Observation gripper differs from home's gripper slot — homing must
        # preserve the observation value, not snap the gripper to home.
        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        observed_gripper = 0.025
        q[gripper_position_idx] = observed_gripper
        obs = _observation_from_q(q)

        action = policy(obs)

        expected_call_count = 1
        assert len(stub_client.homing_calls) == expected_call_count
        expected_target = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        expected_target[gripper_position_idx] = observed_gripper
        np.testing.assert_array_equal(stub_client.homing_calls[0], expected_target)
        # Returns the hold action, not a teleop'd target.
        expected_tool = policy.robot.get_tool_pose(q)
        np.testing.assert_allclose(action.tool_pose.translation, expected_tool.translation)

    def test_y_button_toggles_data_recording(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        stub_client = StubOrchestratorClient()
        policy.orchestrator_client = stub_client

        policy.recording_request = MagicMock(return_value=True)
        policy.recording_status = MagicMock(return_value=RecordingStatus(0, RecordingState.STOPPED))

        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])

        # 1. First press of Y button should start recording
        reader.buttons["Y"] = True
        policy(obs)
        policy.recording_request.assert_called_once_with()
        assert policy.is_recording is False
        policy.recording_status.return_value = RecordingStatus(1, RecordingState.RUNNING)
        assert policy.is_recording is True

        # Reset mock call counts
        policy.recording_request.reset_mock()

        # 2. Holding Y button on the next step should not start or stop recording again (debouncing)
        policy(obs)
        policy.recording_request.assert_not_called()
        assert policy.is_recording is True

        # 3. Release Y button
        reader.buttons["Y"] = False
        policy(obs)
        policy.recording_request.assert_not_called()
        assert policy.is_recording is True

        # 4. Press Y button again should stop recording
        reader.buttons["Y"] = True
        policy(obs)
        policy.recording_request.assert_called_once_with()
        assert policy.is_recording is True
        policy.recording_status.return_value = RecordingStatus(2, RecordingState.STOPPED)
        assert policy.is_recording is False

    def test_homing_target_does_not_mutate_config(self, panda_policy_and_reader):
        """The policy must copy the config arrays before overwriting gripper slots."""
        policy, reader = panda_policy_and_reader
        stub_client = StubOrchestratorClient()
        policy.orchestrator_client = stub_client
        reader.buttons["X"] = True

        original_home = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        q = policy.robot_config.homing_presets[HomingPreset.HOME].copy()
        gripper_position_idx = policy.robot.get_gripper_position_indices()[0]
        q[gripper_position_idx] = 0.025
        policy(_observation_from_q(q))

        np.testing.assert_array_equal(
            policy.robot_config.homing_presets[HomingPreset.HOME], original_home
        )

    def test_homing_passes_through_when_no_gripper_configured(self):
        """With no gripper joints, the target equals the config array verbatim."""
        cfg = replace(ROBOT_CONFIGS["panda"], gripper=None)
        policy = OculusTeleopPolicy(
            robot_config=cfg,
            config=OculusTeleopPolicyConfig(
                subscriptions=TELEOP_SUBSCRIPTIONS,
                verbose=False,
                oculus_to_tool_command_rotation=np.eye(3),
            ),
            orchestrator_client=StubOrchestratorClient(),
        )
        stub_client = StubOrchestratorClient()
        policy.orchestrator_client = stub_client
        policy.reader.buttons["X"] = True  # ty:ignore[unresolved-attribute]

        policy(_observation_from_q(cfg.homing_presets[HomingPreset.HOME]))

        np.testing.assert_array_equal(
            stub_client.homing_calls[0], cfg.homing_presets[HomingPreset.HOME]
        )

    def test_x_and_y_triggered_simultaneously(self, panda_policy_and_reader):
        """When both X and Y are pressed, X triggers homing and Y toggles recording."""
        policy, reader = panda_policy_and_reader
        stub_client = StubOrchestratorClient()
        policy.orchestrator_client = stub_client

        policy.recording_request = MagicMock(return_value=True)
        policy.recording_status = MagicMock(return_value=RecordingStatus(0, RecordingState.STOPPED))
        reader.buttons["X"] = True
        reader.buttons["Y"] = True

        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])
        policy(obs)

        # Check homing triggered
        assert len(stub_client.homing_calls) == 1
        np.testing.assert_array_equal(
            stub_client.homing_calls[0], policy.robot_config.homing_presets[HomingPreset.HOME]
        )

        # Check recording triggered
        policy.recording_request.assert_called_once_with()
        assert policy.is_recording is False
        policy.recording_status.return_value = RecordingStatus(1, RecordingState.RUNNING)
        assert policy.is_recording is True

    def test_no_button_pressed_does_not_request_homing(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        stub_client = StubOrchestratorClient()
        policy.orchestrator_client = stub_client
        reader.buttons["RG"] = True

        policy(_observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME]))

        assert stub_client.homing_calls == []

    def test_x_button_is_edge_triggered(self, panda_policy_and_reader):
        """X button press should trigger homing only once on the rising edge, even if held."""
        policy, reader = panda_policy_and_reader
        stub_client = StubOrchestratorClient()
        policy.orchestrator_client = stub_client
        reader.buttons["X"] = True

        obs = _observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME])

        # 1. First step: button transitioned to pressed, should trigger homing
        policy(obs)
        assert len(stub_client.homing_calls) == ONE_CALL

        # 2. Second step: button is still pressed, should not trigger again
        policy(obs)
        assert len(stub_client.homing_calls) == ONE_CALL

        # 3. Third step: button released
        reader.buttons["X"] = False
        policy(obs)
        assert len(stub_client.homing_calls) == ONE_CALL

        # 4. Fourth step: button pressed again, should trigger again
        reader.buttons["X"] = True
        policy(obs)
        assert len(stub_client.homing_calls) == TWO_CALLS


class TestReset:
    def test_reset_clears_references_and_base(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        policy(_observation_from_q(policy.robot_config.homing_presets[HomingPreset.HOME]))
        assert policy.filtered_controller_pose is not None
        assert policy.reference_base_pose is not None
        assert policy.commanded_gripper_position is not None

        policy.reset()

        assert policy.filtered_controller_pose is None
        assert policy.commanded_tool_pose is None
        assert policy.reference_base_pose is None
        assert policy.commanded_gripper_position is None


def _se3_to_matrix(transform: pin.SE3) -> np.ndarray:
    """4x4 numpy matrix matching the format OculusReader produces."""
    matrix = np.eye(4)
    matrix[:3, :3] = transform.rotation
    matrix[:3, 3] = transform.translation
    return matrix


def _assert_se3_equal(actual: pin.SE3 | None, expected: pin.SE3 | None) -> None:
    assert actual is not None
    assert expected is not None
    np.testing.assert_allclose(actual.rotation, expected.rotation, atol=1e-12)
    np.testing.assert_allclose(actual.translation, expected.translation, atol=1e-12)
