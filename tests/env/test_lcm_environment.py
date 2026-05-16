from unittest.mock import MagicMock, patch

import numpy as np
import pinocchio as pin
import pytest

from humanoid.constants import Topic
from humanoid.environment.lcm import LCMEnvironment
from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)


def _make_state(timestamp: float = 0.0) -> RobotState:
    return RobotState(
        timestamp=timestamp,
        joint_positions=np.arange(7, dtype=float),
        joint_velocities=np.zeros(7),
        motor_temperatures=np.zeros(7),
    )


def _make_env() -> LCMEnvironment:
    """Build an LCMEnvironment with mocked Publisher and Subscriber."""
    with (
        patch("humanoid.environment.lcm.Publisher"),
        patch("humanoid.environment.lcm.Subscriber"),
    ):
        return LCMEnvironment()


@pytest.fixture
def env() -> LCMEnvironment:
    return _make_env()


class TestInit:
    def test_subscribes_to_robot_state_and_joint_command(self):
        with (
            patch("humanoid.environment.lcm.Publisher"),
            patch("humanoid.environment.lcm.Subscriber") as mock_sub,
        ):
            LCMEnvironment()
        mock_sub.assert_called_once_with(
            topics=[
                Topic.ROBOT_STATE,
                Topic.ROBOT_JOINT_COMMAND,
                Topic.ROBOT_TOOL_COMMAND,
                Topic.ROBOT_BASE_COMMAND,
            ]
        )

    def test_default_callbacks_used_when_none_provided(self, env):
        obs = Observation(robot_state=_make_state())
        action = Action()
        assert env.reward_fn(None, action, obs) == 0.0
        assert env.done_fn(obs) is False
        assert env.truncated_fn(obs) is False

    def test_custom_callbacks_are_used(self):
        reward_val = 5.0
        with (
            patch("humanoid.environment.lcm.Publisher"),
            patch("humanoid.environment.lcm.Subscriber"),
        ):
            env = LCMEnvironment(
                reward_fn=lambda prev, a, o: reward_val,
                done_fn=lambda o: True,
                truncated_fn=lambda o: True,
            )
        obs = Observation(robot_state=_make_state())
        assert env.reward_fn(None, Action(), obs) == reward_val
        assert env.done_fn(obs) is True
        assert env.truncated_fn(obs) is True


class TestReset:
    def test_returns_observation_from_robot_state(self, env):
        state = _make_state(timestamp=1.5)
        env.subscriber.receive = MagicMock(side_effect=[state, None, None, None])

        observation = env.reset()

        assert isinstance(observation, Observation)
        assert observation.robot_state is state
        assert observation.robot_joint_command is None
        assert observation.robot_tool_command is None
        assert observation.robot_base_command is None
        calls = env.subscriber.receive.call_args_list
        assert calls[0] == ((Topic.ROBOT_STATE,), {"timeout": env.timeout_ms})
        assert calls[1] == ((Topic.ROBOT_JOINT_COMMAND,), {})
        assert calls[2] == ((Topic.ROBOT_TOOL_COMMAND,), {})
        assert calls[3] == ((Topic.ROBOT_BASE_COMMAND,), {})

    def test_raises_when_no_state_received(self, env):
        env.subscriber.receive = MagicMock(return_value=None)
        with pytest.raises(RuntimeError, match="Failed to receive robot state"):
            env.reset()

    def test_clears_last_action_and_records_prev_observation(self, env):
        env.subscriber.receive = MagicMock(return_value=_make_state())
        env._last_action = Action(joint_positions=np.zeros(7))

        observation = env.reset()

        assert env._last_action is None
        assert env._prev_observation is observation


class TestStep:
    def test_publishes_joint_command_when_joint_positions_set(self, env):
        env.subscriber.receive = MagicMock(return_value=_make_state())
        joint_positions = np.array([0.1, 0.2, 0.3])
        action = Action(joint_positions=joint_positions)

        env.step(action)

        published = env.publisher.publish.call_args_list
        assert len(published) == 1
        cmd = published[0][0][0]
        assert isinstance(cmd, RobotJointCommand)
        np.testing.assert_allclose(cmd.joint_positions, joint_positions)

    def test_publishes_tool_command_when_tool_pose_set(self, env):
        env.subscriber.receive = MagicMock(return_value=_make_state())
        tool_pose = pin.SE3(np.eye(3), np.array([0.3, 0.0, 0.4]))
        gripper = np.array([0.01])
        action = Action(tool_pose=tool_pose, gripper_positions=gripper)

        env.step(action)

        published = env.publisher.publish.call_args_list
        assert len(published) == 1
        cmd = published[0][0][0]
        assert isinstance(cmd, RobotToolCommand)
        np.testing.assert_allclose(cmd.pose.translation, tool_pose.translation)
        np.testing.assert_allclose(cmd.gripper_positions, gripper)

    def test_publishes_base_command_when_base_pose_set(self, env):
        env.subscriber.receive = MagicMock(return_value=_make_state())
        base_pose = pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))
        action = Action(base_pose=base_pose)

        env.step(action)

        published = env.publisher.publish.call_args_list
        assert len(published) == 1
        cmd = published[0][0][0]
        assert isinstance(cmd, RobotBaseCommand)
        np.testing.assert_allclose(cmd.pose.translation, base_pose.translation)

    def test_publishes_multiple_commands_for_combined_action(self, env):
        env.subscriber.receive = MagicMock(return_value=_make_state())
        action = Action(
            joint_positions=np.zeros(7),
            tool_pose=pin.SE3.Identity(),
            base_pose=pin.SE3.Identity(),
        )

        env.step(action)

        published_types = [c.args[0].__class__ for c in env.publisher.publish.call_args_list]
        assert RobotJointCommand in published_types
        assert RobotToolCommand in published_types
        assert RobotBaseCommand in published_types

    def test_empty_action_publishes_nothing(self, env):
        env.subscriber.receive = MagicMock(return_value=_make_state())
        env.step(Action())
        env.publisher.publish.assert_not_called()

    def test_raises_when_no_state_received(self, env):
        env.subscriber.receive = MagicMock(return_value=None)
        with pytest.raises(RuntimeError, match="Failed to receive robot state"):
            env.step(Action())

    def test_callbacks_invoked_with_correct_arguments(self, env):
        prev_state = _make_state(timestamp=1.0)
        prev_obs = Observation(robot_state=prev_state)
        env._prev_observation = prev_obs

        new_state = _make_state(timestamp=2.0)
        env.subscriber.receive = MagicMock(return_value=new_state)

        reward_val = 1.5

        env.reward_fn = MagicMock(return_value=reward_val)
        env.done_fn = MagicMock(return_value=True)
        env.truncated_fn = MagicMock(return_value=False)

        action = Action(joint_positions=np.zeros(7))
        transition = env.step(action)

        # Reward gets (prev_obs, action, new_obs).
        reward_args = env.reward_fn.call_args[0]
        assert reward_args[0] is prev_obs
        assert reward_args[1] is action
        assert reward_args[2].robot_state is new_state

        # Done/truncated get the new observation.
        assert env.done_fn.call_args[0][0].robot_state is new_state
        assert env.truncated_fn.call_args[0][0].robot_state is new_state

        assert transition.reward == reward_val
        assert transition.is_done is True
        assert transition.is_truncated is False

    def test_info_contains_timing_fields(self, env):
        command_timestamp = 40.0
        obs_timestamp = 42.0
        state = _make_state(timestamp=obs_timestamp)
        env.subscriber.receive = MagicMock(return_value=state)

        with patch("humanoid.environment.lcm.time.time", return_value=command_timestamp):
            transition = env.step(Action())

        assert transition.info["command_timestamp"] == command_timestamp
        assert transition.info["observation_timestamp"] == obs_timestamp
        assert transition.info["latency"] == pytest.approx(2.0)

    def test_step_updates_prev_observation_and_last_action(self, env):
        env.subscriber.receive = MagicMock(return_value=_make_state())
        action = Action(joint_positions=np.zeros(7))

        transition = env.step(action)

        assert env._prev_observation is transition.observation
        assert env._last_action is action

    def test_returned_transition_observation_wraps_received_state(self, env):
        state = _make_state(timestamp=3.0)
        env.subscriber.receive = MagicMock(return_value=state)

        transition = env.step(Action())

        assert transition.observation.robot_state is state


class TestClose:
    def test_close_closes_subscriber(self, env):
        env.close()
        env.subscriber.close.assert_called_once()
