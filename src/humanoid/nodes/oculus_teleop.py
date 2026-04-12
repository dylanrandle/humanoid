"""Oculus VR teleoperation node for controlling robot end-effector pose.

This node uses the OculusTeleopPolicy with the LCMEnvironment to provide
Oculus VR controller-based control of the robot's end-effector pose.

Usage:
    uv run python -m humanoid.nodes.oculus_teleop
"""

import sys

from humanoid.config import ROBOT_CONFIG
from humanoid.environment.lcm import LCMEnvironment
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.policy import OculusTeleopPolicy
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class OculusTeleopNode:
    """Node that runs Oculus VR teleoperation using policy and environment.

    This node integrates the OculusTeleopPolicy with the LCMEnvironment
    to provide a clean, modular approach to VR-based robot control.
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        rate_hz: float = DEFAULT_RATE_HZ,
        scale_translation: float = 1.0,
        scale_rotation: float = 1.0,
    ):
        """Initialize the Oculus teleoperation node.

        Args:
            robot_config: Robot configuration
            rate_hz: Control loop rate in Hz
            scale_translation: Scale factor for controller translation
            scale_rotation: Scale factor for controller rotation
        """
        logger.info("Initializing OculusTeleopNode")

        # Create policy with specified settings
        self.policy = OculusTeleopPolicy(
            robot_config=robot_config,
            scale_translation=scale_translation,
            scale_rotation=scale_rotation,
        )

        # Create environment
        self.env = LCMEnvironment()

        self.rate_hz = rate_hz

        logger.info(f"OculusTeleopNode initialized at {rate_hz} Hz")

    def step(self) -> None:
        """Execute one step of the control loop."""
        # Get action from policy
        action = self.policy(self.observation)

        # Execute action in environment
        transition = self.env.step(action)

        # Update observation for next iteration
        self.observation = transition.observation

    def run(self) -> None:
        """Run the Oculus teleoperation node main loop."""
        logger.info(f"Starting Oculus teleop loop at {self.rate_hz} Hz")
        logger.info("Use right controller to control end-effector")
        logger.info("Use right trigger to control gripper\n")

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
        logger.info("Closing OculusTeleopNode...")
        self.env.close()
        logger.info("OculusTeleopNode closed")


def main():
    """Main entry point for the Oculus teleoperation node."""
    node = OculusTeleopNode()
    node.run()


if __name__ == "__main__":
    main()
