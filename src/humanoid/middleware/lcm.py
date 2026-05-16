import contextlib
import queue
import threading
from typing import Literal, overload

import lcm

from humanoid.constants import DEFAULT_LCM_URL, TOPIC_TO_TYPE, TYPE_TO_TOPIC, Topic
from humanoid.logger import get_logger
from humanoid.types.lcm import (
    robot_base_command_t,
    robot_joint_command_t,
    robot_state_t,
    robot_tool_command_t,
)
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

logger = get_logger(__name__)

_NON_BLOCKING_TIMEOUT_MS = 0
_SPIN_TIMEOUT_MS = 100


AcceptedTypes = RobotJointCommand | RobotToolCommand | RobotBaseCommand | RobotState


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
        elif isinstance(data, RobotBaseCommand):
            lcm_data = LCMConverter.robot_base_command_to_lcm(data)
            topic = TYPE_TO_TOPIC[RobotBaseCommand]
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
        queue_size: int | None = 1,
    ):
        self.lc = lcm.LCM(url)
        self.url = url
        self.topics = topics
        self.queue_size = queue_size
        self._subscriptions: list[lcm.LCMSubscription] = []

        maxsize = queue_size if queue_size is not None else 0
        self._message_queues: dict[Topic, queue.Queue[AcceptedTypes]] = {
            topic: queue.Queue(maxsize=maxsize) for topic in topics
        }

        for topic in topics:
            subscription = self.lc.subscribe(topic.value, self._handle_message)
            if queue_size is not None:
                subscription.set_queue_capacity(queue_size)
            self._subscriptions.append(subscription)

        self._running = True
        self._lcm_thread = threading.Thread(target=self._lcm_spin, daemon=True)
        self._lcm_thread.start()

    def _lcm_spin(self) -> None:
        """Background thread that continuously drains the LCM socket."""
        while self._running:
            self.lc.handle_timeout(_SPIN_TIMEOUT_MS)

    def _handle_message(self, channel: str, data: bytes) -> None:
        try:
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
            elif expected_type == RobotBaseCommand:
                lcm_msg = robot_base_command_t.decode(data)
                decoded_data = LCMConverter.robot_base_command_from_lcm(lcm_msg)
            else:
                raise RuntimeError("Encountered unexpected channel")

            q = self._message_queues[topic]
            if q.full():
                with contextlib.suppress(queue.Empty):
                    q.get_nowait()
            q.put_nowait(decoded_data)
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

    @overload
    def receive(
        self, topic: Literal[Topic.ROBOT_BASE_COMMAND], timeout: int | None = None
    ) -> RobotBaseCommand | None: ...

    def receive(self, topic: Topic, timeout: int | None = 0) -> AcceptedTypes | None:
        """Retrieve a message from the per-topic queue.

        Args:
            topic: Which channel to read from.
            timeout: Milliseconds to wait. 0 returns immediately; None blocks forever.
        """
        try:
            if timeout is None:
                return self._message_queues[topic].get(block=True)
            if timeout == _NON_BLOCKING_TIMEOUT_MS:
                return self._message_queues[topic].get(block=False)
            return self._message_queues[topic].get(block=True, timeout=timeout / 1000)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._running = False
        self._lcm_thread.join(timeout=1.0)
        for subscription in self._subscriptions:
            self.lc.unsubscribe(subscription)
        self._subscriptions.clear()
