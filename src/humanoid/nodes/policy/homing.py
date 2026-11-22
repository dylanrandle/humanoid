"""Long-running homing node.

Subscribes to HOMING_TARGET; whenever a new target arrives, the policy retargets
and runs the smooth trajectory. When the trajectory completes the node publishes
an ORCHESTRATOR_EVENT(kind=COMPLETE) so the orchestrator can hand control back
to its previous mode.
"""

import argparse
from dataclasses import replace

from humanoid.config import ROBOT_CONFIG
from humanoid.config.homing import HOMING_POLICY_CONFIG
from humanoid.constants import Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.logger import get_logger
from humanoid.nodes.base import Node
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.policy.homing import HomingPolicy
from humanoid.types.homing import HomingPolicyConfig
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)


class HomingNode(Node):
    """Long-running node that homes the robot whenever a target is received."""

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        policy_config: HomingPolicyConfig = HOMING_POLICY_CONFIG,
    ):
        self.rate_hz = policy_config.rate_hz
        self.policy = HomingPolicy(
            speed=policy_config.speed,
            dt=1.0 / self.rate_hz,
            robot_config=robot_config,
        )
        self.env = RealtimeEnvironment(
            action_topics=ActionTopics(joint=Topic.HOMING_JOINT_COMMAND),
            subscriptions=policy_config.subscriptions,
        )
        self.orchestrator = OrchestratorClient()

        # Whether we've already emitted COMPLETE for the current target.
        self._completed = True

    def setup(self) -> None:
        self.observation = self.env.reset()

    def step(self) -> None:
        target_msg = self.observation.homing_target
        if target_msg is not None:
            logger.info(f"Received homing target ({len(target_msg.target_position)} positions)")
            self.policy.set_target(target_msg.target_position)
            self._completed = False

        action = self.policy(self.observation)
        transition = self.env.step(action)
        self.observation = transition.observation

        if self.policy.is_done and not self._completed and self.observation.homing_target is None:
            logger.info("Homing complete; publishing COMPLETE event")
            self.orchestrator.complete()
            self._completed = True

    def on_close(self) -> None:
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Long-running homing node")
    parser.add_argument(
        "--speed", type=float, default=HOMING_POLICY_CONFIG.speed, help="Max joint speed in rad/s"
    )
    parser.add_argument(
        "--rate", type=float, default=HOMING_POLICY_CONFIG.rate_hz, help="Control rate in Hz"
    )
    args = parser.parse_args()

    HomingNode(
        policy_config=replace(HOMING_POLICY_CONFIG, speed=args.speed, rate_hz=args.rate)
    ).run()


if __name__ == "__main__":
    main()
