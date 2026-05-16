"""Tests for OculusTeleopPolicy.

The ``oculus_reader.OculusReader`` device dependency is patched out with a
mutable stub so we can drive the policy through synthetic controller poses,
button states, and joystick deflections.
"""

from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.policy.teleop import oculus as oculus_module
from humanoid.policy.teleop.oculus import (
    OculusTeleopPolicy,
    OculusTeleopPolicyConfig,
)
from humanoid.types.observation import Observation
from humanoid.types.robot import RobotState


class StubOculusReader:
    """Stand-in for oculus_reader.OculusReader.

    Returns whatever pose/button state the test has loaded into it. Tests
    mutate ``transforms`` and ``buttons`` between policy calls to simulate
    operator input.
    """

    def __init__(self):
        self.transforms: dict = {"r": np.eye(4)}
        self.buttons: dict = {
            "rightTrig": (0.0,),
            "LG": False,
            "RG": False,
            "leftJS": (0.0, 0.0),
            "rightJS": (0.0, 0.0),
        }

    def get_transformations_and_buttons(self):
        return self.transforms, self.buttons


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
            motor_temperatures=np.zeros_like(q),
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
            verbose=False,
            oculus_to_world_rotation=np.eye(3),
        )
    policy = OculusTeleopPolicy(robot_config=robot_config, config=config)
    return policy, policy.reader  # ty:ignore[invalid-return-type]


@pytest.fixture
def panda_policy_and_reader():
    return _make_policy(robot_name="panda")


@pytest.fixture
def mobile_policy_and_reader():
    return _make_policy(robot_name="elrobot_mobile")


class TestConstruction:
    def test_uses_stub_reader(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        assert isinstance(reader, StubOculusReader)
        assert policy.config.verbose is False

    def test_invalid_data_returns_hold_action(self, panda_policy_and_reader):
        """When OculusReader gives empty data the policy must hold position."""
        policy, reader = panda_policy_and_reader
        reader.transforms = {}
        reader.buttons = {}
        obs = _observation_from_q(policy.robot_config.home_position)

        action = policy(obs)

        expected_tool = policy.robot.get_tool_pose(policy.robot_config.home_position)
        np.testing.assert_allclose(action.tool_pose.translation, expected_tool.translation)


class TestDeadManSwitch:
    def test_grip_released_returns_hold_action_and_clears_references(self, panda_policy_and_reader):
        policy, _ = panda_policy_and_reader
        # Pre-populate references to make sure release wipes them.
        policy.reference_controller_pose = np.eye(4)
        policy.reference_tool_pose = pin.SE3.Identity()

        obs = _observation_from_q(policy.robot_config.home_position)
        action = policy(obs)

        assert policy.reference_controller_pose is None
        assert policy.reference_tool_pose is None

        expected_tool = policy.robot.get_tool_pose(policy.robot_config.home_position)
        np.testing.assert_allclose(action.tool_pose.translation, expected_tool.translation)

    @pytest.mark.parametrize(
        "grip_buttons",
        [{"LG": True, "RG": False}, {"LG": False, "RG": True}, {"LG": True, "RG": True}],
    )
    def test_either_grip_engages_motion(self, panda_policy_and_reader, grip_buttons):
        policy, reader = panda_policy_and_reader
        reader.buttons.update(grip_buttons)
        # Move the controller forward (oculus_to_world is identity here, so the
        # controller-frame delta is exactly the world-frame delta).
        reader.transforms = {"r": _se3_to_matrix(pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.0])))}

        obs = _observation_from_q(policy.robot_config.home_position)
        action = policy(obs)

        # References must be set after a single engaged tick.
        assert policy.reference_controller_pose is not None
        assert policy.reference_tool_pose is not None
        assert action.tool_pose is not None


class TestControllerDeltaApplied:
    def test_no_motion_yields_reference_tool_pose(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        # Initial controller pose at identity.
        reader.transforms = {"r": np.eye(4)}

        obs = _observation_from_q(policy.robot_config.home_position)
        action = policy(obs)

        expected = policy.robot.get_tool_pose(policy.robot_config.home_position)
        np.testing.assert_allclose(action.tool_pose.translation, expected.translation, atol=1e-12)

    def test_controller_translation_translates_target(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        # First tick captures the reference (identity).
        policy(_observation_from_q(policy.robot_config.home_position))

        # Move the controller by +0.05m in x.
        delta = np.array([0.05, 0.0, 0.0])
        reader.transforms = {"r": _se3_to_matrix(pin.SE3(np.eye(3), delta))}

        action = policy(_observation_from_q(policy.robot_config.home_position))

        ref = policy.reference_tool_pose
        # ref * SE3(I, delta) — translation = ref.translation + ref.rotation @ delta
        expected_translation = ref.translation + ref.rotation @ delta
        np.testing.assert_allclose(action.tool_pose.translation, expected_translation, atol=1e-12)

    def test_translation_scale_amplifies_delta(self):
        policy, reader = _make_policy(
            config=OculusTeleopPolicyConfig(
                verbose=False,
                scale_translation=3.0,
                oculus_to_world_rotation=np.eye(3),
            ),
        )
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.home_position)
        policy(obs)  # capture reference

        delta = np.array([0.05, 0.0, 0.0])
        reader.transforms = {"r": _se3_to_matrix(pin.SE3(np.eye(3), delta))}
        action = policy(obs)

        ref = policy.reference_tool_pose
        expected_translation = ref.translation + ref.rotation @ (delta * 3.0)  # ty:ignore[unresolved-attribute]
        np.testing.assert_allclose(action.tool_pose.translation, expected_translation, atol=1e-12)  # ty:ignore[unresolved-attribute]

    def test_oculus_to_world_remap_applied_to_delta(self):
        """A 90° rotation about Z should swap x and y axes of the input delta."""
        rotation_about_z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        policy, reader = _make_policy(
            config=OculusTeleopPolicyConfig(
                verbose=False,
                oculus_to_world_rotation=rotation_about_z_90,
            ),
        )
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.home_position)
        policy(obs)

        # +x in the Oculus frame should appear as +y after the remap.
        reader.transforms = {"r": _se3_to_matrix(pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0])))}
        action = policy(obs)

        ref = policy.reference_tool_pose
        remapped_delta = rotation_about_z_90 @ np.array([0.05, 0.0, 0.0])
        expected_translation = ref.translation + ref.rotation @ remapped_delta  # ty:ignore[unresolved-attribute]
        np.testing.assert_allclose(action.tool_pose.translation, expected_translation, atol=1e-12)  # ty:ignore[unresolved-attribute]


class TestGripper:
    def test_trigger_zero_yields_min_position(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        reader.buttons["rightTrig"] = (0.0,)

        action = policy(_observation_from_q(policy.robot_config.home_position))

        assert action.gripper_positions is not None
        assert action.gripper_positions[0] == pytest.approx(policy.gripper_min)

    def test_trigger_one_yields_max_position(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        reader.buttons["rightTrig"] = (1.0,)

        action = policy(_observation_from_q(policy.robot_config.home_position))

        assert action.gripper_positions[0] == pytest.approx(policy.gripper_max)

    def test_no_gripper_indices_yields_none(self):
        cfg = replace(ROBOT_CONFIGS["panda"], gripper_joint_indices=None)
        policy = OculusTeleopPolicy(
            robot_config=cfg,
            config=OculusTeleopPolicyConfig(verbose=False, oculus_to_world_rotation=np.eye(3)),
        )
        policy.reader.buttons["RG"] = True  # ty:ignore[unresolved-attribute]
        policy.reader.buttons["rightTrig"] = (0.5,)  # ty:ignore[unresolved-attribute]

        action = policy(_observation_from_q(cfg.home_position))

        assert action.gripper_positions is None


class TestBasePoseFromJoysticks:
    def test_no_base_frame_yields_no_base_pose(self, panda_policy_and_reader):
        policy, reader = panda_policy_and_reader
        reader.buttons["RG"] = True
        reader.buttons["leftJS"] = (1.0, 1.0)
        reader.buttons["rightJS"] = (1.0, 0.0)

        action = policy(_observation_from_q(policy.robot_config.home_position))

        assert action.base_pose is None

    def test_first_engaged_call_anchors_base_from_fk(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        # Zero stick → no translation/rotation delta.
        obs = _observation_from_q(policy.robot_config.home_position)

        action = policy(obs)

        expected = policy.robot.get_base_pose(policy.robot_config.home_position)
        np.testing.assert_allclose(action.base_pose.translation, expected.translation, atol=1e-12)

    def test_left_stick_forward_drives_base_along_plus_y(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.home_position)
        policy(obs)  # initialize base anchor
        before = policy.reference_base_pose.translation.copy()

        reader.buttons["leftJS"] = (0.0, 1.0)
        policy(obs)

        delta = policy.reference_base_pose.translation - before
        np.testing.assert_allclose(
            delta,
            [0.0, policy.config.base_translation_step, 0.0],
            atol=1e-12,
        )

    def test_left_stick_right_drives_base_along_plus_x(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.home_position)
        policy(obs)
        before = policy.reference_base_pose.translation.copy()

        reader.buttons["leftJS"] = (1.0, 0.0)
        policy(obs)

        delta = policy.reference_base_pose.translation - before
        np.testing.assert_allclose(
            delta,
            [policy.config.base_translation_step, 0.0, 0.0],
            atol=1e-12,
        )

    def test_right_stick_right_yields_negative_yaw(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.home_position)
        policy(obs)
        before = policy.reference_base_pose.rotation.copy()

        reader.buttons["rightJS"] = (1.0, 0.0)
        policy(obs)

        # base_yaw_scale defaults to -1, so +jx -> negative yaw.
        expected_dyaw = policy.config.base_yaw_scale * 1.0 * policy.config.base_rotation_step
        expected = before @ pin.utils.rotate("z", expected_dyaw)
        np.testing.assert_allclose(policy.reference_base_pose.rotation, expected, atol=1e-12)

    def test_joystick_deadzone_suppresses_small_input(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        obs = _observation_from_q(policy.robot_config.home_position)
        policy(obs)
        before = policy.reference_base_pose.translation.copy()

        # Well inside the default deadzone (0.1).
        reader.buttons["leftJS"] = (0.05, 0.05)
        policy(obs)

        np.testing.assert_allclose(policy.reference_base_pose.translation, before, atol=1e-12)


class TestReset:
    def test_reset_clears_references_and_base(self, mobile_policy_and_reader):
        policy, reader = mobile_policy_and_reader
        reader.buttons["RG"] = True
        policy(_observation_from_q(policy.robot_config.home_position))
        assert policy.reference_controller_pose is not None
        assert policy.reference_base_pose is not None

        policy.reset()

        assert policy.reference_controller_pose is None
        assert policy.reference_tool_pose is None
        assert policy.reference_base_pose is None


def _se3_to_matrix(transform: pin.SE3) -> np.ndarray:
    """4x4 numpy matrix matching the format OculusReader produces."""
    matrix = np.eye(4)
    matrix[:3, :3] = transform.rotation
    matrix[:3, 3] = transform.translation
    return matrix
