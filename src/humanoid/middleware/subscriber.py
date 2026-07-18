import contextlib
import queue
import threading
from typing import Literal, overload

import lcm

from humanoid.constants import DEFAULT_LCM_URL, TOPIC_TO_TYPE, Topic
from humanoid.logger import get_logger
from humanoid.types.homing import HomingTarget
from humanoid.types.lcm import (
    homing_target_t,
    logging_status_t,
    orchestrator_event_t,
    orchestrator_mode_t,
    robot_base_command_t,
    robot_joint_command_t,
    robot_state_t,
    robot_tool_command_t,
)
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.logging import LoggingStatus
from humanoid.types.middleware import AcceptedTypes
from humanoid.types.orchestrator import OrchestratorEvent, OrchestratorMode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

logger = get_logger(__name__)

_NON_BLOCKING_TIMEOUT_MS = 0
_SPIN_TIMEOUT_MS = 100


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

            if expected_type is RobotJointCommand:
                lcm_msg = robot_joint_command_t.decode(data)
                decoded_data = LCMConverter.robot_joint_command_from_lcm(lcm_msg)
            elif expected_type is RobotState:
                lcm_msg = robot_state_t.decode(data)
                decoded_data = LCMConverter.robot_state_from_lcm(lcm_msg)
            elif expected_type is RobotToolCommand:
                lcm_msg = robot_tool_command_t.decode(data)
                decoded_data = LCMConverter.robot_tool_command_from_lcm(lcm_msg)
            elif expected_type is RobotBaseCommand:
                lcm_msg = robot_base_command_t.decode(data)
                decoded_data = LCMConverter.robot_base_command_from_lcm(lcm_msg)
            elif expected_type is OrchestratorMode:
                lcm_msg = orchestrator_mode_t.decode(data)
                decoded_data = LCMConverter.orchestrator_mode_from_lcm(lcm_msg)
            elif expected_type is OrchestratorEvent:
                lcm_msg = orchestrator_event_t.decode(data)
                decoded_data = LCMConverter.orchestrator_event_from_lcm(lcm_msg)
            elif expected_type is HomingTarget:
                lcm_msg = homing_target_t.decode(data)
                decoded_data = LCMConverter.homing_target_from_lcm(lcm_msg)
            elif expected_type is LoggingStatus:
                lcm_msg = logging_status_t.decode(data)
                decoded_data = LCMConverter.logging_status_from_lcm(lcm_msg)
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
        self,
        topic: Literal[
            Topic.ROBOT_JOINT_COMMAND,
            Topic.CONTROLLER_JOINT_COMMAND,
            Topic.HOMING_JOINT_COMMAND,
        ],
        timeout: int | None = None,
    ) -> RobotJointCommand | None: ...

    @overload
    def receive(
        self, topic: Literal[Topic.ROBOT_STATE], timeout: int | None = None
    ) -> RobotState | None: ...

    @overload
    def receive(
        self,
        topic: Literal[
            Topic.ROBOT_TOOL_COMMAND,
            Topic.OCULUS_TOOL_COMMAND,
            Topic.KEYBOARD_TOOL_COMMAND,
        ],
        timeout: int | None = None,
    ) -> RobotToolCommand | None: ...

    @overload
    def receive(
        self,
        topic: Literal[
            Topic.ROBOT_BASE_COMMAND,
            Topic.OCULUS_BASE_COMMAND,
            Topic.KEYBOARD_BASE_COMMAND,
        ],
        timeout: int | None = None,
    ) -> RobotBaseCommand | None: ...

    @overload
    def receive(
        self, topic: Literal[Topic.ORCHESTRATOR_MODE], timeout: int | None = None
    ) -> OrchestratorMode | None: ...

    @overload
    def receive(
        self, topic: Literal[Topic.ORCHESTRATOR_EVENT], timeout: int | None = None
    ) -> OrchestratorEvent | None: ...

    @overload
    def receive(
        self, topic: Literal[Topic.HOMING_TARGET], timeout: int | None = None
    ) -> HomingTarget | None: ...

    @overload
    def receive(
        self, topic: Literal[Topic.LOGGING_STATUS], timeout: int | None = None
    ) -> LoggingStatus | None: ...

    @overload
    def receive(self, topic: Topic, timeout: int | None = None) -> AcceptedTypes | None: ...

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
