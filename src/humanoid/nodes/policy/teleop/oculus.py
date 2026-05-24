"""Oculus VR teleoperation node for controlling robot end-effector pose.

This node uses the OculusTeleopPolicy with the RealtimeEnvironment to provide
Oculus VR controller-based control of the robot's end-effector pose.
"""

from humanoid.config import ROBOT_CONFIG
from humanoid.environment.realtime import RealtimeEnvironment
from humanoid.logger import get_logger
from humanoid.nodes.base import Node
from humanoid.policy import OculusTeleopPolicy, OculusTeleopPolicyConfig
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)


class OculusTeleopNode(Node):
    """Node that runs Oculus VR teleoperation using policy and environment.

    This node integrates the OculusTeleopPolicy with the RealtimeEnvironment
    to provide a clean, modular approach to VR-based robot control.
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        policy_config: OculusTeleopPolicyConfig | None = None,
    ):
        """Initialize the Oculus teleoperation node.

        Args:
            robot_config: Robot configuration
            policy_config: Tunable parameters for the Oculus teleop policy.
                Its ``dt`` field sets the control loop period.
        """
        if policy_config is None:
            policy_config = OculusTeleopPolicyConfig()

        self.policy = OculusTeleopPolicy(robot_config=robot_config, config=policy_config)

        # Create environment
        self.env = RealtimeEnvironment()

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
