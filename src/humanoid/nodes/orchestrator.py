"""Orchestrator node: selects the active policy by routing per-source command
topics to the final ROBOT_* topics consumed by the robot driver and OSC.

For each mode, a static map declares which per-source topics get forwarded to
which final topics. The orchestrator also broadcasts the active Mode so the
OSC can re-sync from ROBOT_STATE when it is not the active source.
"""

import argparse
import time

from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.types.orchestrator import Mode, OrchestratorMode

logger = get_logger(__name__)

# NOTE: rate needs to be high enough to not bottleneck upstream producers.
DEFAULT_RATE_HZ = 500.0

# For each mode, the per-source -> final topic forwards that should be active.
MODE_FORWARDS: dict[Mode, dict[Topic, Topic]] = {
    Mode.IDLE: {},
    Mode.HOMING: {
        Topic.HOMING_JOINT_COMMAND: Topic.ROBOT_JOINT_COMMAND,
    },
    Mode.OCULUS: {
        Topic.OCULUS_TOOL_COMMAND: Topic.ROBOT_TOOL_COMMAND,
        Topic.OCULUS_BASE_COMMAND: Topic.ROBOT_BASE_COMMAND,
        Topic.CONTROLLER_JOINT_COMMAND: Topic.ROBOT_JOINT_COMMAND,
    },
    Mode.KEYBOARD: {
        Topic.KEYBOARD_TOOL_COMMAND: Topic.ROBOT_TOOL_COMMAND,
        Topic.KEYBOARD_BASE_COMMAND: Topic.ROBOT_BASE_COMMAND,
        Topic.CONTROLLER_JOINT_COMMAND: Topic.ROBOT_JOINT_COMMAND,
    },
}


def _all_source_topics() -> list[Topic]:
    """Union of every per-source topic across all modes."""
    topics: set[Topic] = set()
    for forwards in MODE_FORWARDS.values():
        topics.update(forwards.keys())
    return sorted(topics, key=lambda t: t.value)


class OrchestratorNode(Node):
    """Routes per-source policy/controller outputs to the final ROBOT_* topics."""

    def __init__(self, mode: Mode = Mode.IDLE, rate_hz: float = DEFAULT_RATE_HZ):
        self.rate_hz = rate_hz
        self.mode = mode

        self._source_topics = _all_source_topics()
        self.subscriber = Subscriber(topics=self._source_topics)
        self.publisher = Publisher()

    def setup(self) -> None:
        logger.info(f"Orchestrator starting in mode: {self.mode}")

    def step(self) -> None:
        # Broadcast current mode every tick so subscribers stay in sync even if
        # they (re)start mid-session.
        self.publisher.publish(
            OrchestratorMode(timestamp=time.time(), mode=self.mode),
            topic=Topic.ORCHESTRATOR_MODE,
        )

        forwards = MODE_FORWARDS[self.mode]
        for source_topic in self._source_topics:
            msg = self.subscriber.receive(source_topic)
            if msg is None:
                continue
            dest_topic = forwards.get(source_topic)
            if dest_topic is None:
                # Inactive source — discard so its queue doesn't keep stale
                # messages that would leak through on mode change.
                continue
            self.publisher.publish(msg, topic=dest_topic)

    def on_close(self) -> None:
        self.subscriber.close()


def main():
    parser = argparse.ArgumentParser(description="Route per-source commands to the robot")
    parser.add_argument(
        "-m",
        "--mode",
        type=Mode,
        choices=list(Mode),
        default=Mode.IDLE,
        help="Initial active mode (selects which per-source topics get forwarded)",
    )
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RATE_HZ, help="Orchestrator loop rate in Hz"
    )
    args = parser.parse_args()

    OrchestratorNode(mode=args.mode, rate_hz=args.rate).run()


if __name__ == "__main__":
    main()
