"""Keyboard teleoperation node for controlling robot end-effector pose.

This node uses the KeyboardTeleopPolicy with the LCMEnvironment to provide
keyboard-based control of the robot's end-effector pose.

Usage:
    uv run python -m humanoid.nodes.keyboard_teleop
"""

import sys

from humanoid.config import ROBOT_CONFIG
from humanoid.environment.lcm import LCMEnvironment
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.policy.keyboard_teleop import KeyboardTeleopPolicy
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class KeyboardTeleopNode:
    """Node that runs keyboard teleoperation using policy and environment.

    This node integrates the KeyboardTeleopPolicy with the LCMEnvironment
    to provide a clean, modular approach to keyboard-based robot control.
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        """Initialize the keyboard teleoperation node.

        Args:
            robot_config: Robot configuration
            rate_hz: Control loop rate in Hz
            timeout_ms: Timeout in milliseconds for receiving robot state
        """
        logger.info("Initializing KeyboardTeleopNode")

        # Create policy with default settings
        self.policy = KeyboardTeleopPolicy(robot_config=robot_config)

        # Create environment
        self.env = LCMEnvironment()

        self.rate_hz = rate_hz

        logger.info(f"KeyboardTeleopNode initialized at {rate_hz} Hz")

    def step(self) -> None:
        """Execute one step of the control loop."""
        # Get action from policy
        action = self.policy(self.observation)

        # Execute action in environment
        transition = self.env.step(action)

        # Update observation for next iteration
        self.observation = transition.observation

    def run(self) -> None:
        """Run the keyboard teleoperation node main loop."""
        logger.info(f"Starting keyboard teleop loop at {self.rate_hz} Hz")
        logger.info("Press ESC or 'x' to quit\n")

        try:
            # Reset environment and get initial observation
            self.observation = self.env.reset()

            # Run control loop
            loop_at_rate(self.step, rate_hz=self.rate_hz)

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            sys.exit(1)
        finally:
            self.close()

    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing KeyboardTeleopNode...")
        self.policy.stop_listener()
        self.env.close()
        logger.info("KeyboardTeleopNode closed")


def main():
    """Main entry point for the keyboard teleoperation node."""
    node = KeyboardTeleopNode()
    node.run()


if __name__ == "__main__":
    main()
