"""Long-running homing node.

Subscribes to HOMING_TARGET; whenever a new target arrives, the policy retargets
and runs the smooth trajectory. When the trajectory completes the node publishes
an ORCHESTRATOR_EVENT(kind=COMPLETE) so the orchestrator can hand control back
to its previous mode.
"""

import argparse

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.logger import get_logger
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.policy.homing import HomingPolicy
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class HomingNode(Node):
    """Long-running node that homes the robot whenever a target is received."""

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        speed: float = 1.0,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        self.rate_hz = rate_hz
        self.policy = HomingPolicy(
            speed=speed,
            dt=1.0 / rate_hz,
            robot_config=robot_config,
        )
        self.env = RealtimeEnvironment(action_topics=ActionTopics(joint=Topic.HOMING_JOINT_COMMAND))
        self.target_subscriber = Subscriber(topics=[Topic.HOMING_TARGET])
        self.orchestrator = OrchestratorClient()

        # Whether we've already emitted COMPLETE for the current target.
        self._completed = True

    def setup(self) -> None:
        self.observation = self.env.reset()

    def step(self) -> None:
        target_msg = self.target_subscriber.receive(Topic.HOMING_TARGET)
        if target_msg is not None:
            logger.info(f"Received homing target ({len(target_msg.target_position)} positions)")
            self.policy.set_target(target_msg.target_position)
            self._completed = False

        action = self.policy(self.observation)
        transition = self.env.step(action)
        self.observation = transition.observation

        if self.policy.is_done and not self._completed:
            logger.info("Homing complete; publishing COMPLETE event")
            self.orchestrator.complete()
            self._completed = True

    def on_close(self) -> None:
        self.target_subscriber.close()
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Long-running homing node")
    parser.add_argument("--speed", type=float, default=1.0, help="Max joint speed in rad/s")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="Control rate in Hz")
    args = parser.parse_args()

    HomingNode(speed=args.speed, rate_hz=args.rate).run()


if __name__ == "__main__":
    main()
