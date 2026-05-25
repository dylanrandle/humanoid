"""Keyboard teleoperation node for controlling robot end-effector pose.

This node uses the KeyboardTeleopPolicy with the RealtimeEnvironment to provide
keyboard-based control of the robot's end-effector pose.
"""

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.logger import get_logger
from humanoid.nodes.base import Node
from humanoid.policy import KeyboardTeleopPolicy, KeyboardTeleopPolicyConfig
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)


class KeyboardTeleopNode(Node):
    """Node that runs keyboard teleoperation using policy and environment.

    This node integrates the KeyboardTeleopPolicy with the RealtimeEnvironment
    to provide a clean, modular approach to keyboard-based robot control.
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        policy_config: KeyboardTeleopPolicyConfig | None = None,
    ):
        """Initialize the keyboard teleoperation node.

        Args:
            robot_config: Robot configuration
            policy_config: Tunable parameters for the keyboard teleop policy.
                Its ``dt`` field sets the control loop period.
        """
        if policy_config is None:
            policy_config = KeyboardTeleopPolicyConfig()

        self.policy = KeyboardTeleopPolicy(robot_config=robot_config, config=policy_config)

        self.env = RealtimeEnvironment(
            action_topics=ActionTopics(
                tool=Topic.KEYBOARD_TOOL_COMMAND,
                base=Topic.KEYBOARD_BASE_COMMAND,
            )
        )

        self.rate_hz = 1.0 / policy_config.dt

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
