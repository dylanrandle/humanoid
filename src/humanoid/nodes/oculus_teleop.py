"""Oculus VR teleoperation node for controlling robot end-effector pose.

This node uses the OculusTeleopPolicy with the LCMEnvironment to provide
Oculus VR controller-based control of the robot's end-effector pose.

Usage:
    uv run python -m humanoid.nodes.oculus_teleop
"""

from humanoid.config import ROBOT_CONFIG
from humanoid.environment.lcm import LCMEnvironment
from humanoid.logger import get_logger
from humanoid.nodes.base import Node
from humanoid.policy import OculusTeleopPolicy, OculusTeleopPolicyConfig
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class OculusTeleopNode(Node):
    """Node that runs Oculus VR teleoperation using policy and environment.

    This node integrates the OculusTeleopPolicy with the LCMEnvironment
    to provide a clean, modular approach to VR-based robot control.
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        policy_config: OculusTeleopPolicyConfig | None = None,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        """Initialize the Oculus teleoperation node.

        Args:
            robot_config: Robot configuration
            policy_config: Tunable parameters for the Oculus teleop policy
            rate_hz: Control loop rate in Hz
        """

        self.policy = OculusTeleopPolicy(robot_config=robot_config, config=policy_config)

        # Create environment
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
        logger.info("Hold either grip trigger (dead-man) to command motion")
        logger.info("Use right controller pose to control end-effector")
        logger.info("Use right trigger to control gripper")
        logger.info("Use left joystick to drive the base (forward = y+, right = x+)")
        logger.info("Use right joystick X to yaw the base (left = yaw+, right = yaw-)\n")
        self.observation = self.env.reset()

    def on_close(self) -> None:
        self.env.close()


def main():
    """Main entry point for the Oculus teleoperation node."""
    node = OculusTeleopNode()
    node.run()


if __name__ == "__main__":
    main()
