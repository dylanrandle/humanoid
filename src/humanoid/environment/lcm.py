import time
from collections.abc import Callable
from typing import Any

from humanoid.constants import Topic
from humanoid.environment.base import Environment
from humanoid.logger import get_logger
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.types.action import Action
from humanoid.types.observation import Observation
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


class LCMEnvironment(Environment):
    """Environment implementation using LCM for robot communication.

    This environment:
    - Subscribes to ROBOT_STATE topic to receive observations
    - Publishes to ROBOT_JOINT_COMMAND or ROBOT_TOOL_COMMAND based on action type
    - Implements the standard Environment interface with reset() and step()
    """

    def __init__(
        self,
        timeout_ms: int = 100,
        reward_fn: RewardFunction | None = None,
        done_fn: DoneFunction | None = None,
        truncated_fn: TruncatedFunction | None = None,
    ):
        """Initialize the LCM environment.

        Args:
            timeout_ms: Timeout in milliseconds for receiving messages
            reward_fn: Optional function to compute reward
            done_fn: Optional function to determine if episode is done
            truncated_fn: Optional function to determine if episode is truncated
        """
        self.timeout_ms = timeout_ms

        # Initialize LCM communication
        self.publisher = Publisher()
        self.subscriber = Subscriber(topics=[Topic.ROBOT_STATE], queue_size=1)

        # Reward and termination functions
        self.reward_fn = reward_fn or self._default_reward
        self.done_fn = done_fn or self._default_done
        self.truncated_fn = truncated_fn or self._default_truncated

        # Track previous observation for reward computation
        self._prev_observation: Observation | None = None
        self._last_action: Action | None = None

        logger.info("Initialized LCMEnvironment")

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Observation:
        """Reset the environment to an initial state.

        Args:
            seed: Optional random seed (not used in LCM environment)
            options: Optional dictionary of environment-specific options

        Returns:
            Initial observation after reset
        """
        logger.info("Resetting environment")

        # Wait for a fresh robot state
        robot_state = self._receive_robot_state()

        if robot_state is None:
            raise RuntimeError("Failed to receive robot state during reset")

        observation = Observation(robot_state=robot_state)
        self._prev_observation = observation
        self._last_action = None

        return observation

    def step(self, action: Action) -> Transition:
        """Execute one step in the environment.

        Args:
            action: Action to execute (joint or tool space)

        Returns:
            Transition containing observation, reward, is_done, is_truncated, and info
        """
        # Send command based on action type
        timestamp = time.time()

        if action.joint_positions is not None:
            command = RobotJointCommand(
                timestamp=timestamp,
                joint_positions=action.joint_positions,
            )
            self.publisher.publish(command)
            logger.debug("Published joint command")

        if action.tool_pose is not None:
            command = RobotToolCommand(
                timestamp=timestamp,
                pose=action.tool_pose,
                gripper_positions=action.gripper_positions,
            )
            self.publisher.publish(command)
            logger.debug("Published tool command")

        if action.base_pose is not None:
            base_command = RobotBaseCommand(timestamp=timestamp, pose=action.base_pose)
            self.publisher.publish(base_command)
            logger.debug("Published base command")

        # Receive next observation
        robot_state = self._receive_robot_state()

        if robot_state is None:
            raise RuntimeError("Failed to receive robot state after action")

        observation = Observation(robot_state=robot_state)

        # Compute reward, done, and truncated
        reward = self.reward_fn(self._prev_observation, action, observation)
        is_done = self.done_fn(observation)
        is_truncated = self.truncated_fn(observation)

        # Create info dict with timing information
        info: TransitionInfo = {
            "command_timestamp": timestamp,
            "observation_timestamp": robot_state.timestamp,
            "latency": robot_state.timestamp - timestamp,
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
        logger.info("Closing LCM environment")
        self.subscriber.close()

    def _receive_robot_state(self) -> RobotState | None:
        """Receive a robot state from LCM.

        Returns:
            RobotState if received, None if timeout
        """
        return self.subscriber.receive(Topic.ROBOT_STATE, timeout=self.timeout_ms)

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
