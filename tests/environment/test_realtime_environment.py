from dataclasses import replace
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.homing import HOMING_POLICY_CONFIG
from humanoid.constants import Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.types.action import Action
from humanoid.types.camera import CameraEncoding, CameraImage, CameraIntrinsics, camera_topics
from humanoid.types.homing import HomingTarget
from humanoid.types.observation import Observation, ObservationSubscriptions
from humanoid.types.orchestrator import Mode, OrchestratorMode
from humanoid.types.robot import (
    CartesianVelocity,
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

_ALL_FEEDBACK = ObservationSubscriptions(
    topics=(
        Topic.ROBOT_STATE,
        Topic.ROBOT_JOINT_COMMAND,
        Topic.ROBOT_TOOL_COMMAND,
        Topic.ROBOT_BASE_COMMAND,
        Topic.ORCHESTRATOR_MODE,
    ),
    camera_names=("base", "wrist"),
)

_TEST_TOPICS = ActionTopics(
    joint=Topic.HOMING_JOINT_COMMAND,
    tool=Topic.OCULUS_TOOL_COMMAND,
    base=Topic.OCULUS_BASE_COMMAND,
)


def _make_state(timestamp: float = 0.0) -> RobotState:
    return RobotState(
        timestamp=timestamp,
        joint_positions=np.arange(7, dtype=float),
        joint_velocities=np.zeros(7),
        actuator_temperatures=np.zeros(7),
    )


def _state_only_receive(state: RobotState):
    """Return ``state`` for ROBOT_STATE and None for every other topic."""

    def receive(topic, timeout=0):
        if topic == Topic.ROBOT_STATE:
            return state
        return None

    return receive


def _make_env(
    action_topics: ActionTopics = _TEST_TOPICS,
    subscriptions: ObservationSubscriptions = _ALL_FEEDBACK,
) -> RealtimeEnvironment:
    """Build a RealtimeEnvironment with mocked Publisher and Subscriber."""
    with (
        patch("humanoid.environment.realtime.Publisher"),
        patch("humanoid.environment.realtime.Subscriber"),
    ):
        return RealtimeEnvironment(action_topics=action_topics, subscriptions=subscriptions)


@pytest.fixture
def env() -> RealtimeEnvironment:
    return _make_env()


class TestInit:
    def test_subscribes_to_robot_state_and_final_command_topics(self):
        with (
            patch("humanoid.environment.realtime.Publisher"),
            patch("humanoid.environment.realtime.Subscriber") as mock_sub,
        ):
            RealtimeEnvironment(action_topics=_TEST_TOPICS, subscriptions=_ALL_FEEDBACK)
        mock_sub.assert_called_once_with(
            topics=[
                Topic.ROBOT_STATE,
                Topic.ROBOT_JOINT_COMMAND,
                Topic.ROBOT_TOOL_COMMAND,
                Topic.ROBOT_BASE_COMMAND,
                Topic.ORCHESTRATOR_MODE,
                camera_topics("base").rgb,
                camera_topics("base").depth,
                camera_topics("base").registered_depth,
                camera_topics("wrist").rgb,
                camera_topics("wrist").depth,
                camera_topics("wrist").registered_depth,
            ],
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
            patch("humanoid.environment.realtime.Publisher"),
            patch("humanoid.environment.realtime.Subscriber"),
        ):
            env = RealtimeEnvironment(
                action_topics=_TEST_TOPICS,
                subscriptions=_ALL_FEEDBACK,
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
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(state))

        observation = env.reset()

        assert isinstance(observation, Observation)
        assert observation.robot_state is state
        assert observation.robot_joint_command is None
        assert observation.robot_tool_command is None
        assert observation.robot_base_command is None
        assert observation.mode is None
        calls = env.subscriber.receive.call_args_list
        assert calls[0] == ((Topic.ROBOT_STATE,), {"timeout": env.timeout_ms})
        assert calls[1] == ((Topic.ROBOT_JOINT_COMMAND,), {"timeout": env.timeout_ms})
        assert calls[2] == ((Topic.ROBOT_TOOL_COMMAND,), {"timeout": env.timeout_ms})
        assert calls[3] == ((Topic.ROBOT_BASE_COMMAND,), {"timeout": env.timeout_ms})
        assert calls[4] == ((Topic.ORCHESTRATOR_MODE,), {"timeout": env.timeout_ms})

    def test_observation_includes_orchestrator_mode(self, env):
        state = _make_state()
        mode_msg = OrchestratorMode(timestamp=0.0, mode=Mode.OCULUS)
        env.subscriber.receive = MagicMock(
            side_effect=lambda topic, **_: {
                Topic.ROBOT_STATE: state,
                Topic.ORCHESTRATOR_MODE: mode_msg,
            }.get(topic)
        )

        observation = env.reset()

        assert observation.mode is Mode.OCULUS

    def test_raises_when_no_state_received(self, env):
        env.subscriber.receive = MagicMock(return_value=None)
        with pytest.raises(RuntimeError, match="Failed to receive robot state"):
            env.reset()

    def test_falls_back_to_cached_state_when_receive_returns_none(self, env):
        state = _make_state(timestamp=1.0)
        # First reset populates the cache.
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(state))
        env.reset()

        # Second reset: ROBOT_STATE returns None — should use cached state.
        env.subscriber.receive = MagicMock(return_value=None)
        observation = env.reset()

        assert observation.robot_state is state

    def test_falls_back_to_cached_commands_when_receive_returns_none(self, env):
        state = _make_state()
        joint_cmd = RobotJointCommand(timestamp=0.0, joint_positions=np.zeros(7))
        # First reset populates the caches.
        env.subscriber.receive = MagicMock(
            side_effect=lambda topic, **_: {
                Topic.ROBOT_STATE: state,
                Topic.ROBOT_JOINT_COMMAND: joint_cmd,
            }.get(topic)
        )
        env.reset()

        # Second reset: command topics return None — cached joint_cmd should be reused.
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(state))
        observation = env.reset()

        assert observation.robot_joint_command is joint_cmd

    def test_clears_last_action_and_records_prev_observation(self, env):
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(_make_state()))
        env._last_action = Action(joint_positions=np.zeros(7))

        observation = env.reset()

        assert env._last_action is None
        assert env._prev_observation is observation


class TestStep:
    def test_publishes_joint_command_to_configured_topic(self, env):
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(_make_state()))
        joint_positions = np.array([0.1, 0.2, 0.3])
        action = Action(joint_positions=joint_positions)

        env.step(action)

        published = env.publisher.publish.call_args_list
        assert len(published) == 1
        cmd = published[0].args[0]
        assert isinstance(cmd, RobotJointCommand)
        np.testing.assert_allclose(cmd.joint_positions, joint_positions)
        assert published[0].kwargs["topic"] is Topic.HOMING_JOINT_COMMAND

    def test_publishes_tool_command_to_configured_topic(self, env):
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(_make_state()))
        tool_pose = pin.SE3(np.eye(3), np.array([0.3, 0.0, 0.4]))
        gripper = np.array([0.01])
        tool_velocity = CartesianVelocity(
            linear=np.array([0.1, 0.0, 0.0]),
            angular=np.array([0.0, 0.0, 0.2]),
        )
        action = Action(
            tool_pose=tool_pose,
            tool_velocity=tool_velocity,
            gripper_positions=gripper,
        )

        env.step(action)

        published = env.publisher.publish.call_args_list
        assert len(published) == 1
        cmd = published[0].args[0]
        assert isinstance(cmd, RobotToolCommand)
        np.testing.assert_allclose(cmd.pose.translation, tool_pose.translation)
        np.testing.assert_allclose(cmd.gripper_positions, gripper)
        assert cmd.velocity is tool_velocity
        assert published[0].kwargs["topic"] is Topic.OCULUS_TOOL_COMMAND

    def test_publishes_base_command_to_configured_topic(self, env):
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(_make_state()))
        base_pose = pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))
        action = Action(base_pose=base_pose)

        env.step(action)

        published = env.publisher.publish.call_args_list
        assert len(published) == 1
        cmd = published[0].args[0]
        assert isinstance(cmd, RobotBaseCommand)
        np.testing.assert_allclose(cmd.pose.translation, base_pose.translation)
        assert published[0].kwargs["topic"] is Topic.OCULUS_BASE_COMMAND

    def test_publishes_multiple_commands_for_combined_action(self, env):
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(_make_state()))
        action = Action(
            joint_positions=np.zeros(7),
            tool_pose=pin.SE3.Identity(),
            base_pose=pin.SE3.Identity(),
        )

        env.step(action)

        published_pairs = [
            (c.args[0].__class__, c.kwargs["topic"]) for c in env.publisher.publish.call_args_list
        ]
        assert (RobotJointCommand, Topic.HOMING_JOINT_COMMAND) in published_pairs
        assert (RobotToolCommand, Topic.OCULUS_TOOL_COMMAND) in published_pairs
        assert (RobotBaseCommand, Topic.OCULUS_BASE_COMMAND) in published_pairs

    def test_empty_action_publishes_nothing(self, env):
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(_make_state()))
        env.step(Action())
        env.publisher.publish.assert_not_called()

    def test_raises_when_action_field_set_but_topic_missing(self):
        env = _make_env(action_topics=ActionTopics(tool=Topic.OCULUS_TOOL_COMMAND))
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(_make_state()))  # ty:ignore[invalid-assignment]
        action = Action(joint_positions=np.zeros(7))

        with pytest.raises(RuntimeError, match=r"action_topics\.joint is None"):
            env.step(action)

    def test_raises_when_no_state_received(self, env):
        env.subscriber.receive = MagicMock(return_value=None)
        with pytest.raises(RuntimeError, match="Failed to receive robot state"):
            env.step(Action())

    def test_callbacks_invoked_with_correct_arguments(self, env):
        prev_state = _make_state(timestamp=1.0)
        prev_obs = Observation(robot_state=prev_state)
        env._prev_observation = prev_obs

        new_state = _make_state(timestamp=2.0)
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(new_state))

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
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(state))

        with patch(
            "humanoid.environment.realtime.time.perf_counter",
            return_value=command_timestamp,
        ):
            transition = env.step(Action())

        assert transition.info["command_timestamp"] == command_timestamp
        assert transition.info["observation_timestamp"] == obs_timestamp
        assert transition.info["latency"] == pytest.approx(2.0)

    def test_step_updates_prev_observation_and_last_action(self, env):
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(_make_state()))
        action = Action(joint_positions=np.zeros(7))

        transition = env.step(action)

        assert env._prev_observation is transition.observation
        assert env._last_action is action

    def test_returned_transition_observation_wraps_received_state(self, env):
        state = _make_state(timestamp=3.0)
        env.subscriber.receive = MagicMock(side_effect=_state_only_receive(state))

        transition = env.step(Action())

        assert transition.observation.robot_state is state


class TestClose:
    def test_close_closes_subscriber(self, env):
        env.close()
        env.subscriber.close.assert_called_once()


def test_optional_camera_observations_pair_by_host_timestamp():
    with (
        patch("humanoid.environment.realtime.Publisher"),
        patch("humanoid.environment.realtime.Subscriber"),
    ):
        environment = RealtimeEnvironment(_TEST_TOPICS, subscriptions=_ALL_FEEDBACK)
    intrinsics = CameraIntrinsics(
        width=1,
        height=1,
        fx=1,
        fy=1,
        cx=0,
        cy=0,
        distortion_model="none",
        distortion_coefficients=(),
    )
    rgb = CameraImage(
        timestamp=1,
        device_timestamp=1,
        timestamp_domain="test",
        frame_number=1,
        frame_id="rgb",
        serial_number="camera",
        encoding=CameraEncoding.RGB8,
        data=np.zeros((1, 1, 3), dtype=np.uint8),
        intrinsics=intrinsics,
    )
    depth = replace(
        rgb,
        frame_id="depth",
        encoding=CameraEncoding.DEPTH16,
        data=np.ones((1, 1), dtype=np.uint16),
        depth_scale=0.001,
        timestamp=2,
    )
    messages = {
        Topic.ROBOT_STATE: _make_state(),
        camera_topics("base").rgb: rgb,
        camera_topics("base").depth: depth,
    }
    cast(MagicMock, environment.subscriber).receive.side_effect = lambda topic, **_: messages.pop(
        topic, None
    )
    assert environment.observe().camera_frames.get("base") is None
    messages[camera_topics("base").rgb] = replace(rgb, timestamp=2)
    paired = environment.observe().camera_frames.get("base")
    assert paired is not None and paired.depth is depth
    messages[camera_topics("base").rgb] = replace(rgb, timestamp=3)
    assert environment.observe().camera_frames.get("base") is paired


def test_subscription_selection_controls_both_subscriber_and_reads():
    wrist = camera_topics("wrist")
    subscriptions = ObservationSubscriptions(
        topics=(Topic.ROBOT_STATE, Topic.ROBOT_BASE_COMMAND),
        camera_names=("wrist",),
    )
    with (
        patch("humanoid.environment.realtime.Publisher"),
        patch("humanoid.environment.realtime.Subscriber") as make_subscriber,
    ):
        env = RealtimeEnvironment(_TEST_TOPICS, subscriptions=subscriptions)
    selected = [*subscriptions.topics, wrist.rgb, wrist.depth, wrist.registered_depth]
    assert make_subscriber.call_args.kwargs["topics"] == selected
    base = RobotBaseCommand(timestamp=1.0, pose=pin.SE3.Identity())
    messages = {Topic.ROBOT_STATE: _make_state(), Topic.ROBOT_BASE_COMMAND: base}

    def receive(topic, **kwargs):
        assert topic in selected
        return messages.pop(topic, None)

    make_subscriber.return_value.receive.side_effect = receive
    for obs in (env.reset(), env.observe(), env.step(Action()).observation):
        assert obs.robot_base_command is base
        assert (
            obs.mode is None and obs.robot_tool_command is None and obs.robot_joint_command is None
        )
        assert obs.camera_frames == {}


def test_homing_target_is_delivered_once_per_publication():
    env = _make_env(subscriptions=HOMING_POLICY_CONFIG.subscriptions)
    target = HomingTarget(timestamp=1.0, target_position=np.zeros(7))
    messages = {Topic.ROBOT_STATE: _make_state(), Topic.HOMING_TARGET: target}
    cast(MagicMock, env.subscriber).receive.side_effect = lambda topic, **_: messages.pop(
        topic, None
    )
    assert env.reset().homing_target is target
    assert env.step(Action()).observation.homing_target is None
    messages[Topic.HOMING_TARGET] = target
    assert env.step(Action()).observation.homing_target is target
