"""Keyboard teleoperation node for controlling robot end-effector pose.

This node uses the KeyboardTeleopPolicy with the LCMEnvironment to provide
keyboard-based control of the robot's end-effector pose.

Usage:
    uv run python -m humanoid.nodes.keyboard_teleop
"""

from humanoid.config import ROBOT_CONFIG
from humanoid.environment.lcm import LCMEnvironment
from humanoid.logger import get_logger
from humanoid.nodes.base import Node
from humanoid.policy import KeyboardTeleopPolicy, KeyboardTeleopPolicyConfig
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class KeyboardTeleopNode(Node):
    """Node that runs keyboard teleoperation using policy and environment.

    This node integrates the KeyboardTeleopPolicy with the LCMEnvironment
    to provide a clean, modular approach to keyboard-based robot control.
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        policy_config: KeyboardTeleopPolicyConfig | None = None,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        """Initialize the keyboard teleoperation node.

        Args:
            robot_config: Robot configuration
            policy_config: Tunable parameters for the keyboard teleop policy
            rate_hz: Control loop rate in Hz
        """

        self.policy = KeyboardTeleopPolicy(robot_config=robot_config, config=policy_config)

        self.env = LCMEnvironment()

        self.rate_hz = rate_hz

    def step(self) -> None:
        """Execute one step of the control loop."""
        # Get action from policy
        action = self.policy(self.observation)

        # Execute action in environment
        transition = self.env.step(action)

        # Update observation for next iteration
        self.observation = transition.observation

    def setup(self) -> None:
        logger.info("Press ESC or 'x' to quit\n")
        self.observation = self.env.reset()

    def on_close(self) -> None:
        self.policy.stop_listener()
        self.env.close()


def main():
    """Main entry point for the keyboard teleoperation node."""
    node = KeyboardTeleopNode()
    node.run()


if __name__ == "__main__":
    main()
