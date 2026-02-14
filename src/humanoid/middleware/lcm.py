from typing import Literal, overload

import lcm

from humanoid.constants import DEFAULT_LCM_URL, TOPIC_TO_TYPE, TYPE_TO_TOPIC, Topic
from humanoid.logger import get_logger
from humanoid.types.lcm import robot_joint_command_t, robot_state_t, robot_tool_command_t
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import RobotJointCommand, RobotState, RobotToolCommand

logger = get_logger(__name__)


AcceptedTypes = RobotJointCommand | RobotToolCommand | RobotState


class Publisher:
    def __init__(self, url: str = DEFAULT_LCM_URL):
        self.lc = lcm.LCM(url)
        self.url = url

    def publish(self, data: AcceptedTypes) -> None:
        # Convert to LCM type based on data type
        if isinstance(data, RobotJointCommand):
            lcm_data = LCMConverter.robot_joint_command_to_lcm(data)
            topic = TYPE_TO_TOPIC[RobotJointCommand]
        elif isinstance(data, RobotState):
            lcm_data = LCMConverter.robot_state_to_lcm(data)
            topic = TYPE_TO_TOPIC[RobotState]
        elif isinstance(data, RobotToolCommand):
            lcm_data = LCMConverter.robot_tool_command_to_lcm(data)
            topic = TYPE_TO_TOPIC[RobotToolCommand]
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Encode and publish
        data_bytes = lcm_data.encode()
        self.lc.publish(topic.value, data_bytes)


class Subscriber:
    def __init__(
        self,
        topics: list[Topic],
        url: str = DEFAULT_LCM_URL,
    ):
        self.lc = lcm.LCM(url)
        self.url = url
        self.topics = topics
        self._subscriptions: list[lcm.LCMSubscription] = []
        self._message_queues: dict[Topic, list[AcceptedTypes]] = {}

        for topic in topics:
            subscription = self.lc.subscribe(topic.value, self._handle_message)
            self._subscriptions.append(subscription)
            self._message_queues[topic] = []

    def _handle_message(self, channel: str, data: bytes) -> None:
        try:
            # Determine expected type from channel name
            topic = Topic(channel)
            expected_type = TOPIC_TO_TYPE.get(topic)

            if expected_type == RobotJointCommand:
                lcm_msg = robot_joint_command_t.decode(data)
                decoded_data = LCMConverter.robot_joint_command_from_lcm(lcm_msg)
            elif expected_type == RobotState:
                lcm_msg = robot_state_t.decode(data)
                decoded_data = LCMConverter.robot_state_from_lcm(lcm_msg)
            elif expected_type == RobotToolCommand:
                lcm_msg = robot_tool_command_t.decode(data)
                decoded_data = LCMConverter.robot_tool_command_from_lcm(lcm_msg)
            else:
                raise RuntimeError("Encountered unexpected channel")

            self._message_queues[topic].append(decoded_data)
        except Exception as e:
            logger.error(f"Error decoding message on channel {channel}: {e}")

    @overload
    def receive(
        self, topic: Literal[Topic.ROBOT_JOINT_COMMAND], timeout: int | None = None
    ) -> RobotJointCommand | None: ...

    @overload
    def receive(
        self, topic: Literal[Topic.ROBOT_STATE], timeout: int | None = None
    ) -> RobotState | None: ...

    @overload
    def receive(
        self, topic: Literal[Topic.ROBOT_TOOL_COMMAND], timeout: int | None = None
    ) -> RobotToolCommand | None: ...

    def receive(self, topic: Topic, timeout: int | None = None) -> AcceptedTypes | None:
        # Check if we have queued messages for this topic
        if self._message_queues[topic]:
            return self._message_queues[topic].pop(0)

        # Handle LCM messages
        if timeout is None:
            # Blocking receive
            self.lc.handle()
        else:
            self.lc.handle_timeout(int(timeout))

        # Return queued message if available for this topic
        if self._message_queues[topic]:
            return self._message_queues[topic].pop(0)

        return None

    def close(self) -> None:
        for subscription in self._subscriptions:
            self.lc.unsubscribe(subscription)
        self._subscriptions.clear()
