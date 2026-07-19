"""Tests for BaseTeleopPolicy shared logic."""

from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.policy.teleop.base import BaseTeleopPolicy
from humanoid.types.action import Action
from humanoid.types.homing import HomingPreset
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotState


class _DummyTeleopPolicy(BaseTeleopPolicy):
    """Concrete subclass so we can instantiate the otherwise-abstract base."""

    mode = Mode.OCULUS

    def step(self, observation: Observation) -> Action:
        return self._hold_current_pose_action(observation)


def _observation_from_q(q: np.ndarray) -> Observation:
    """Build a minimal Observation whose joint_positions equal q."""
    state = RobotState(
        timestamp=0.0,
        joint_positions=q.copy(),
        joint_velocities=np.zeros_like(q),
        actuator_temperatures=np.zeros_like(q),
    )
    return Observation(robot_state=state)


@pytest.fixture(scope="module")
def panda_config():
    return ROBOT_CONFIGS["panda"]


@pytest.fixture(scope="module")
def mobile_config():
    return ROBOT_CONFIGS["elrobot_mobile"]


@pytest.fixture
def panda_policy(panda_config) -> _DummyTeleopPolicy:
    return _DummyTeleopPolicy(robot_config=panda_config, verbose=False)


@pytest.fixture
def mobile_policy(mobile_config) -> _DummyTeleopPolicy:
    return _DummyTeleopPolicy(robot_config=mobile_config, verbose=False)


class TestConstruction:
    def test_gripper_limits_pulled_from_model(self, panda_policy, panda_config):
        expected = panda_policy.robot.get_gripper_limits()[0]
        assert (panda_policy.gripper_min, panda_policy.gripper_max) == expected
        assert panda_policy.gripper_max > panda_policy.gripper_min
        # Sanity: matches the underlying URDF position limit directly.
        pos_idx = panda_policy.robot.get_gripper_position_indices()[0]
        assert panda_policy.gripper_min == pytest.approx(
            panda_policy.robot.model.lowerPositionLimit[pos_idx]
        )
        assert panda_policy.gripper_max == pytest.approx(
            panda_policy.robot.model.upperPositionLimit[pos_idx]
        )

    def test_no_gripper_config_zeroes_limits(self, panda_config):
        no_gripper = replace(panda_config, gripper_joint_indices=None)
        policy = _DummyTeleopPolicy(robot_config=no_gripper, verbose=False)
        assert policy.gripper_min == 0.0
        assert policy.gripper_max == 0.0

    def test_multi_gripper_config_rejected(self, panda_config):
        multi = replace(panda_config, gripper_joint_indices=[7, 7])
        with pytest.raises(AssertionError, match="only supports 1 gripper joint"):
            _DummyTeleopPolicy(robot_config=multi, verbose=False)

    def test_robot_uses_the_provided_config(self, panda_policy, panda_config):
        assert panda_policy.robot_config is panda_config
        assert panda_policy.robot.config is panda_config


class TestForwardKinematicsHelpers:
    def test_tool_pose_matches_robot_fk(self, panda_policy, panda_config):
        obs = _observation_from_q(panda_config.homing_presets[HomingPreset.HOME])
        result = panda_policy._get_current_tool_pose(obs)
        expected = panda_policy.robot.get_tool_pose(panda_config.homing_presets[HomingPreset.HOME])
        np.testing.assert_allclose(result.translation, expected.translation)
        np.testing.assert_allclose(result.rotation, expected.rotation)

    def test_base_pose_is_none_for_fixed_base_robot(self, panda_policy, panda_config):
        obs = _observation_from_q(panda_config.homing_presets[HomingPreset.HOME])
        assert panda_policy._get_current_base_pose(obs) is None

    def test_base_pose_returned_for_mobile_robot(self, mobile_policy, mobile_config):
        obs = _observation_from_q(mobile_config.homing_presets[HomingPreset.HOME])
        result = mobile_policy._get_current_base_pose(obs)
        assert isinstance(result, pin.SE3)
        expected = mobile_policy.robot.get_base_pose(
            mobile_config.homing_presets[HomingPreset.HOME]
        )
        np.testing.assert_allclose(result.translation, expected.translation)


class TestGripperFromObservation:
    def test_returns_none_when_no_grippers_configured(self, panda_config):
        no_gripper = replace(panda_config, gripper_joint_indices=None)
        policy = _DummyTeleopPolicy(robot_config=no_gripper, verbose=False)
        obs = _observation_from_q(panda_config.homing_presets[HomingPreset.HOME])
        assert policy._get_current_gripper_positions(obs) is None

    def test_reads_via_position_index_not_joint_index(self, mobile_policy, mobile_config):
        """Regression: gripper read must use position index, not joint index.

        On the mobile robot the planar base joint shifts the q layout, so the
        gripper joint index (11) differs from its position index in q (17).
        We craft a synthetic q where those two slots hold distinct values and
        confirm the helper returns the position-index slot.
        """
        joint_idx = mobile_config.gripper_joint_indices[0]
        position_idx = mobile_policy.robot.joint_idx_to_position_idx(joint_idx)
        assert joint_idx != position_idx, (
            "Test premise: mobile-robot gripper joint_idx must differ from position_idx."
        )

        q = mobile_config.homing_presets[HomingPreset.HOME].copy()
        gripper_value = 0.123
        decoy_value = 0.987
        q[position_idx] = gripper_value
        q[joint_idx] = decoy_value

        result = mobile_policy._get_current_gripper_positions(_observation_from_q(q))
        assert result is not None
        np.testing.assert_allclose(result, [gripper_value])


class TestHoldCurrentPoseAction:
    def test_panda_hold_action_has_no_base_and_correct_gripper(self, panda_policy, panda_config):
        q = panda_config.homing_presets[HomingPreset.HOME]
        action = panda_policy._hold_current_pose_action(_observation_from_q(q))

        assert isinstance(action, Action)
        assert action.base_pose is None

        expected_tool = panda_policy.robot.get_tool_pose(q)
        np.testing.assert_allclose(action.tool_pose.translation, expected_tool.translation)

        gripper_pos_idx = panda_policy.robot.get_gripper_position_indices()[0]
        np.testing.assert_allclose(action.gripper_positions, [q[gripper_pos_idx]])

    def test_mobile_hold_action_includes_base(self, mobile_policy, mobile_config):
        q = mobile_config.homing_presets[HomingPreset.HOME].copy()
        root_q_slice = mobile_policy.robot.get_root_q_slice()
        assert root_q_slice is not None
        q[root_q_slice][0] = 1.0

        action = mobile_policy._hold_current_pose_action(_observation_from_q(q))

        assert action.base_pose is not None
        expected_base = mobile_policy.robot.get_base_pose(q)
        expected_tool = mobile_policy.robot.get_tool_command_pose(q)
        world_tool = mobile_policy.robot.get_tool_pose(q)
        np.testing.assert_allclose(action.base_pose.translation, expected_base.translation)
        np.testing.assert_allclose(action.tool_pose.translation, expected_tool.translation)
        assert not np.allclose(action.tool_pose.translation, world_tool.translation)
