import argparse
import sys
import time

from humanoid.config import ROBOT_CONFIG
from humanoid.environment.lcm import LCMEnvironment
from humanoid.logger import get_logger
from humanoid.policy.homing import HomingPolicy
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class HomingNode:
    """Node that moves the robot to a target joint configuration."""

    def __init__(
        self,
        target_position,
        robot_config: RobotConfig = ROBOT_CONFIG,
        speed: float = 1.0,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        logger.info(f"Initializing HomingNode for: {robot_config.name}")
        self.rate_hz = rate_hz
        self.policy = HomingPolicy(
            target_position=target_position,
            speed=speed,
            dt=1.0 / rate_hz,
        )
        self.env = LCMEnvironment()
        logger.info(f"HomingNode initialized at {rate_hz} Hz")

    def step(self) -> None:
        action = self.policy(self.observation)
        transition = self.env.step(action)
        self.observation = transition.observation

    def run(self) -> None:
        logger.info(f"Starting homing at {self.rate_hz} Hz")
        try:
            self.observation = self.env.reset()
            period = 1.0 / self.rate_hz
            while not self.policy.is_done:
                t0 = time.perf_counter()
                self.step()
                sleep_time = period - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            logger.info("Homing complete")
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            sys.exit(1)
        finally:
            self.close()

    def close(self) -> None:
        logger.info("Closing HomingNode...")
        self.env.close()
        logger.info("HomingNode closed")


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
