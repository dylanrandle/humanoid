"""Quest VR teleoperation node for controlling robot end-effector pose.

This node uses the QuestTeleopPolicy with the RealtimeEnvironment to provide
Quest VR controller-based control of the robot's end-effector pose.
"""

from humanoid.api.clients.recording import RecordingClient
from humanoid.config import ROBOT_CONFIG, get_quest_teleop_policy_config
from humanoid.constants import Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.logger import get_logger
from humanoid.nodes.base import Node
from humanoid.policy import QuestTeleopPolicy, QuestTeleopPolicyConfig
from humanoid.types.robot import RobotConfig

logger = get_logger(__name__)


class QuestTeleopNode(Node):
    """Node that runs Quest VR teleoperation using policy and environment.

    This node integrates the QuestTeleopPolicy with the RealtimeEnvironment
    to provide a clean, modular approach to VR-based robot control.
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        policy_config: QuestTeleopPolicyConfig | None = None,
    ):
        """Initialize the Quest teleoperation node.

        Args:
            robot_config: Robot configuration
            policy_config: Tunable parameters for the Quest teleop policy.
                Its ``dt`` field sets the control loop period.
        """
        if policy_config is None:
            policy_config = get_quest_teleop_policy_config()
        self.recording = RecordingClient()
        self.policy = QuestTeleopPolicy(
            robot_config=robot_config,
            config=policy_config,
            recording_request=self.recording.request,
            recording_status=self.recording.snapshot,
        )

        # Create environment
        self.env = RealtimeEnvironment(
            action_topics=ActionTopics(
                tool=Topic.QUEST_TOOL_COMMAND,
                base=Topic.QUEST_BASE_COMMAND,
            ),
            subscriptions=policy_config.subscriptions,
        )

        self.rate_hz = 1.0 / self.policy.config.dt

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
        logger.info("Hold A to close or B to open the gripper")
        logger.info("Press X to move home and Y to toggle recording")
        logger.info("Use left joystick to drive the base (forward = y+, right = x+)")
        logger.info("Use right joystick X to yaw the base (left = yaw+, right = yaw-)\n")
        self.observation = self.env.reset()

    def on_close(self) -> None:
        try:
            self.policy.input_client.close()
        finally:
            self.recording.close()
            self.env.close()


def main():
    """Main entry point for the Quest teleoperation node."""
    node = QuestTeleopNode()
    node.run()


if __name__ == "__main__":
    main()
