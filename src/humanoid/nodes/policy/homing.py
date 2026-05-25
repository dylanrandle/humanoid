import argparse

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.logger import get_logger
from humanoid.nodes.base import Node
from humanoid.policy.homing import HomingPolicy
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class HomingNode(Node):
    """Node that moves the robot to a target joint configuration."""

    def __init__(
        self,
        target_position,
        robot_config: RobotConfig = ROBOT_CONFIG,
        speed: float = 1.0,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        self.rate_hz = rate_hz
        self.policy = HomingPolicy(
            target_position=target_position,
            speed=speed,
            dt=1.0 / rate_hz,
        )
        self.env = RealtimeEnvironment(action_topics=ActionTopics(joint=Topic.HOMING_JOINT_COMMAND))

    def setup(self) -> None:
        self.observation = self.env.reset()

    def step(self) -> None:
        action = self.policy(self.observation)
        transition = self.env.step(action)
        self.observation = transition.observation

    def stop_condition(self) -> bool:
        return self.policy.is_done

    def on_close(self) -> None:
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Move robot to a target joint configuration")
    parser.add_argument(
        "-p",
        "--position",
        type=str,
        required=True,
        choices=["home", "rest"],
        help="Target position: 'home' or 'rest'",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Max joint speed in rad/s")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="Control rate in Hz")
    args = parser.parse_args()

    target = {"home": ROBOT_CONFIG.home_position, "rest": ROBOT_CONFIG.rest_position}[args.position]
    HomingNode(target_position=target, speed=args.speed, rate_hz=args.rate).run()


if __name__ == "__main__":
    main()
