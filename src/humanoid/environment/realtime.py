import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from humanoid.constants import Topic
from humanoid.environment.base import Environment
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.types.action import Action
from humanoid.types.camera import (
    CameraFrames,
    CameraImage,
    CameraTopics,
    RegisteredDepth,
    camera_topics,
)
from humanoid.types.observation import Observation, ObservationSubscriptions
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.transition import Transition, TransitionInfo

logger = get_logger(__name__)


# Type aliases for callback functions
RewardFunction = Callable[[Observation | None, Action, Observation], float]
DoneFunction = Callable[[Observation], bool]
TruncatedFunction = Callable[[Observation], bool]


@dataclass
class ActionTopics:
    """Per-source topics each action field is published to.

    A policy that only emits joint commands sets ``joint``; a teleop policy that
    emits tool/base commands sets ``tool`` and ``base``. A field left as None
    must not appear on actions submitted to this environment.
    """

    joint: Topic | None = None
    tool: Topic | None = None
    base: Topic | None = None


@dataclass
class _CameraStream:
    topics: CameraTopics
    rgb: CameraImage | None = None
    depth: CameraImage | None = None
    registered_depth: RegisteredDepth | None = None
    camera_frames: CameraFrames | None = None

    def read(self, subscriber: Subscriber) -> CameraFrames | None:
        rgb = subscriber.receive(self.topics.rgb)
        depth = subscriber.receive(self.topics.depth)
        if rgb is not None:
            self.rgb = rgb
        if depth is not None:
            self.depth = depth
        if (
            self.rgb is not None
            and self.depth is not None
            and self.rgb.timestamp == self.depth.timestamp
            and self.rgb.serial_number == self.depth.serial_number
        ):
            incoming = subscriber.receive(self.topics.registered_depth)
            if incoming is not None:
                self.registered_depth = incoming
            registered = self.registered_depth
            self.camera_frames = CameraFrames(
                rgb=self.rgb,
                depth=self.depth,
                registered=registered
                if registered and registered.matches(self.rgb, self.depth)
                else None,
            )
        return self.camera_frames


class RealtimeEnvironment(Environment):
    """Environment that talks to a running robot (or simulator) over LCM.

    This environment:
    - Subscribes to the feedback topics and named cameras requested by the policy.
    - Publishes each action field to its configured per-source topic in
      ``action_topics``. The orchestrator decides which per-source topic
      currently feeds each final ROBOT_* topic.
    """

    def __init__(  # noqa: PLR0913 - transport and reward functions are independent
        self,
        action_topics: ActionTopics,
        timeout_ms: int = 100,
        reward_fn: RewardFunction | None = None,
        done_fn: DoneFunction | None = None,
        truncated_fn: TruncatedFunction | None = None,
        *,
        subscriptions: ObservationSubscriptions,
    ):
        """Initialize the environment.

        Args:
            action_topics: Per-source topics to publish each action field on.
            subscriptions: Robot feedback, events and camera streams to observe.
            timeout_ms: Timeout in milliseconds for receiving messages on reset
            reward_fn: Optional function to compute reward
            done_fn: Optional function to determine if episode is done
            truncated_fn: Optional function to determine if episode is truncated
        """
        self.action_topics = action_topics
        self.timeout_ms = timeout_ms

        # Initialize LCM communication
        self.subscriptions = subscriptions
        self._cameras = {
            name: _CameraStream(camera_topics(name)) for name in subscriptions.camera_names
        }
        self.publisher = Publisher()
        self.subscriber = Subscriber(
            topics=[
                *subscriptions.topics,
                *(
                    topic
                    for stream in self._cameras.values()
                    for topic in (
                        stream.topics.rgb,
                        stream.topics.depth,
                        stream.topics.registered_depth,
                    )
                ),
            ],
        )

        # Reward and termination functions
        self.reward_fn = reward_fn or self._default_reward
        self.done_fn = done_fn or self._default_done
        self.truncated_fn = truncated_fn or self._default_truncated

        # Track previous observation for reward computation
        self._prev_observation: Observation | None = None
        self._last_action: Action | None = None

        # Cache of most recently received messages; used as fallback when receive() returns None
        self._last_robot_state: RobotState | None = None
        self._last_joint_command: RobotJointCommand | None = None
        self._last_tool_command: RobotToolCommand | None = None
        self._last_base_command: RobotBaseCommand | None = None
        self._last_mode: Mode | None = None

        logger.info("Initialized environment")

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        *,
        state_timeout_ms: int | None = None,
    ) -> Observation:
        """Reset the environment to an initial state.

        Args:
            seed: Optional random seed (not used in LCM environment)
            options: Optional dictionary of environment-specific options
            state_timeout_ms: Override the startup wait for the first robot state.

        Returns:
            Initial observation after reset
        """
        logger.info("Resetting environment")

        observation = self._build_observation(
            timeout_ms=self.timeout_ms, state_timeout_ms=state_timeout_ms
        )
        self._prev_observation = observation
        self._last_action = None

        logger.info("Environment reset!")

        return observation

    def observe(self) -> Observation:
        """Read available asynchronous state without issuing an action."""
        return self._build_observation()

    def step(self, action: Action) -> Transition:
        """Execute one step in the environment.

        Args:
            action: Action to execute (joint or tool space)

        Returns:
            Transition containing observation, reward, is_done, is_truncated, and info
        """
        timestamp = self._submit_action(action)
        observation = self._build_observation()

        # Compute reward, done, and truncated
        reward = self.reward_fn(self._prev_observation, action, observation)
        is_done = self.done_fn(observation)
        is_truncated = self.truncated_fn(observation)

        # Create info dict with timing information
        info: TransitionInfo = {
            "command_timestamp": timestamp,
            "observation_timestamp": observation.robot_state.timestamp,
            "latency": observation.robot_state.timestamp - timestamp,
        }

        # Update state
        self._prev_observation = observation
        self._last_action = action

        return Transition(
            observation=observation,
            reward=reward,
            is_done=is_done,
            is_truncated=is_truncated,
            info=info,
        )

    def close(self) -> None:
        """Clean up environment resources."""
        logger.info("Closing environment")
        self.subscriber.close()

    def _submit_action(self, action: Action) -> float:
        """Publish commands for each non-None field in action.

        Returns:
            Timestamp (seconds) at which the commands were published
        """
        # Robot state and controller output use the monotonic performance clock;
        # command timestamps share that clock so latency remains meaningful.
        timestamp = time.perf_counter()

        if action.joint_positions is not None:
            if self.action_topics.joint is None:
                raise RuntimeError("action.joint_positions set but action_topics.joint is None")
            self.publisher.publish(
                RobotJointCommand(timestamp=timestamp, joint_positions=action.joint_positions),
                topic=self.action_topics.joint,
            )
            logger.debug("Published joint command")

        if action.tool_pose is not None:
            if self.action_topics.tool is None:
                raise RuntimeError("action.tool_pose set but action_topics.tool is None")
            self.publisher.publish(
                RobotToolCommand(
                    timestamp=timestamp,
                    pose=action.tool_pose,
                    gripper_positions=action.gripper_positions,
                    velocity=action.tool_velocity,
                ),
                topic=self.action_topics.tool,
            )
            logger.debug("Published tool command")

        if action.base_pose is not None:
            if self.action_topics.base is None:
                raise RuntimeError("action.base_pose set but action_topics.base is None")
            self.publisher.publish(
                RobotBaseCommand(timestamp=timestamp, pose=action.base_pose),
                topic=self.action_topics.base,
            )
            logger.debug("Published base command")

        return timestamp

    def _build_observation(
        self, timeout_ms: int = 0, state_timeout_ms: int | None = None
    ) -> Observation:
        """Receive the latest messages and build an Observation.

        Retain the last feedback sample from each configured stream. Events are
        delivered once. Raises if robot state has never been received.

        Raises:
            RuntimeError: If no robot state has ever been received
        """
        robot_state = self.subscriber.receive(
            Topic.ROBOT_STATE, timeout=timeout_ms if state_timeout_ms is None else state_timeout_ms
        )
        if robot_state is not None:
            self._last_robot_state = robot_state
        if self._last_robot_state is None:
            raise RuntimeError("Failed to receive robot state")

        homing_target = None
        for topic in self.subscriptions.topics:
            match topic:
                case Topic.ROBOT_JOINT_COMMAND:
                    self._last_joint_command = (
                        self.subscriber.receive(topic, timeout=timeout_ms)
                        or self._last_joint_command
                    )
                case Topic.ROBOT_TOOL_COMMAND:
                    self._last_tool_command = (
                        self.subscriber.receive(topic, timeout=timeout_ms)
                        or self._last_tool_command
                    )
                case Topic.ROBOT_BASE_COMMAND:
                    self._last_base_command = (
                        self.subscriber.receive(topic, timeout=timeout_ms)
                        or self._last_base_command
                    )
                case Topic.ORCHESTRATOR_MODE:
                    mode = self.subscriber.receive(topic, timeout=timeout_ms)
                    if mode is not None:
                        self._last_mode = mode.mode
                case Topic.HOMING_TARGET:
                    # Retarget once per request, even when the target is identical.
                    homing_target = self.subscriber.receive(topic)

        return Observation(
            robot_state=self._last_robot_state,
            robot_joint_command=self._last_joint_command,
            robot_tool_command=self._last_tool_command,
            robot_base_command=self._last_base_command,
            mode=self._last_mode,
            homing_target=homing_target,
            camera_frames={
                name: frames
                for name, stream in self._cameras.items()
                if (frames := stream.read(self.subscriber)) is not None
            },
        )

    @staticmethod
    def _default_reward(
        prev_observation: Observation | None,
        action: Action,
        observation: Observation,
    ) -> float:
        """Default reward function (returns 0).

        Override this by passing a custom reward_fn to __init__.
        """
        return 0.0

    @staticmethod
    def _default_done(observation: Observation) -> bool:
        """Default done function (returns False).

        Override this by passing a custom done_fn to __init__.
        """
        return False

    @staticmethod
    def _default_truncated(observation: Observation) -> bool:
        """Default truncated function (returns False).

        Override this by passing a custom truncated_fn to __init__.
        """
        return False
