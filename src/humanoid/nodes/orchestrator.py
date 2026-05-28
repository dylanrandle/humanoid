"""Orchestrator node: an event-driven FSM that selects the active control mode.

External actors (CLI, teleop, etc.) push events to ORCHESTRATOR_EVENT. The
orchestrator transitions between modes accordingly, broadcasts the current
mode on ORCHESTRATOR_MODE, and forwards per-source command topics to the
final ROBOT_* topics for the active mode.

A small "return mode" register implements the canonical
``teleop -> homing -> teleop`` flow: requesting homing from a teleop mode
saves the previous mode and pops back to it when the homing policy sends a
``COMPLETE`` event.
"""

import argparse
import time

from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.types.orchestrator import (
    EventKind,
    Mode,
    OrchestratorEvent,
    OrchestratorMode,
)

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

# Modes the orchestrator will pop back to after a transient HOMING. Requesting
# HOMING from any other mode leaves the return slot empty, so COMPLETE falls
# back to IDLE.
_RETURNABLE_MODES = {Mode.OCULUS, Mode.KEYBOARD}


def _all_source_topics() -> list[Topic]:
    """Union of every per-source topic across all modes."""
    topics: set[Topic] = set()
    for forwards in MODE_FORWARDS.values():
        topics.update(forwards.keys())
    return sorted(topics, key=lambda t: t.value)


class OrchestratorNode(Node):
    """Event-driven FSM that selects the active mode and routes per-source topics."""

    def __init__(self, mode: Mode = Mode.IDLE, rate_hz: float = DEFAULT_RATE_HZ):
        self.rate_hz = rate_hz
        self.mode = mode
        # Where to return after HOMING completes. None outside of HOMING.
        self.return_mode: Mode | None = None

        self._source_topics = _all_source_topics()
        self.subscriber = Subscriber(
            topics=[*self._source_topics, Topic.ORCHESTRATOR_EVENT],
        )
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

        # Drain incoming events; each may change self.mode.
        while True:
            event = self.subscriber.receive(Topic.ORCHESTRATOR_EVENT)
            if event is None:
                break
            self._handle_event(event)

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

    def _handle_event(self, event: OrchestratorEvent) -> None:
        kind = event.kind
        if kind is EventKind.REQUEST_HOMING:
            # Save the prior mode so we can return to it after HOMING.
            if self.mode is not Mode.HOMING and self.mode in _RETURNABLE_MODES:
                self.return_mode = self.mode
            self._transition_to(Mode.HOMING)
        elif kind is EventKind.REQUEST_OCULUS:
            self.return_mode = None
            self._transition_to(Mode.OCULUS)
        elif kind is EventKind.REQUEST_KEYBOARD:
            self.return_mode = None
            self._transition_to(Mode.KEYBOARD)
        elif kind is EventKind.REQUEST_IDLE:
            self.return_mode = None
            self._transition_to(Mode.IDLE)
        elif kind is EventKind.COMPLETE:
            next_mode = self.return_mode or Mode.IDLE
            self.return_mode = None
            self._transition_to(next_mode)
        elif kind in {EventKind.START_LOGGING, EventKind.STOP_LOGGING}:
            logger.info(f"Orchestrator received logging event: {kind}")

    def _transition_to(self, new_mode: Mode) -> None:
        if new_mode is self.mode:
            return
        logger.info(f"Orchestrator: {self.mode} -> {new_mode}")
        self.mode = new_mode

    def on_close(self) -> None:
        self.subscriber.close()


def main():
    parser = argparse.ArgumentParser(description="Event-driven orchestrator FSM")
    parser.add_argument(
        "-m",
        "--mode",
        type=Mode,
        choices=list(Mode),
        default=Mode.IDLE,
        help="Initial mode (events received later override this)",
    )
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RATE_HZ, help="Orchestrator loop rate in Hz"
    )
    args = parser.parse_args()

    OrchestratorNode(mode=args.mode, rate_hz=args.rate).run()


if __name__ == "__main__":
    main()
