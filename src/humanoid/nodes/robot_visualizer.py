"""Robot visualizer node that displays the robot state in real-time.

This node subscribes to the ROBOT_STATE LCM channel and visualizes the robot's
current joint positions using MeshCat.
"""

from humanoid.config import ROBOT_CONFIG, VISUALIZER_CONFIG
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.middleware.lcm import Subscriber
from humanoid.robots.base import Robot
from humanoid.types.robot import RobotConfig
from humanoid.types.visualizer import VisualizerConfig
from humanoid.visualizers.meshcat import MeshcatVisualizer

logger = get_logger(__name__)


DEFAULT_RATE_HZ = 30.0


class RobotVisualizer:
    """Visualizer node that displays robot state in real-time.

    This node subscribes to the ROBOT_STATE, ROBOT_JOINT_COMMAND, and
    ROBOT_TOOL_COMMAND topics and updates a MeshCat visualization with the
    robot's current joint positions, commanded joint positions, and commanded
    tool pose.

    Attributes:
        robot: Robot instance for kinematics and visualization
        viz: MeshCat visualizer instance
        subscriber: LCM subscriber for robot state messages
        running: Flag to control the main loop
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        visualizer_config: VisualizerConfig = VISUALIZER_CONFIG,
    ):
        logger.info(f"Initializing RobotVisualizer for: {robot_config.name}")

        # Load robot model
        self.robot = Robot(robot_config)

        # Setup MeshCat visualization
        self.viz = MeshcatVisualizer(self.robot, config=visualizer_config)
        self.viz.initialize()

        # Initialize with home position
        self.current_q = robot_config.home_position.copy()
        self.viz.display(self.current_q)

        # Setup LCM subscriber
        self.subscriber = Subscriber(
            [Topic.ROBOT_STATE, Topic.ROBOT_JOINT_COMMAND, Topic.ROBOT_TOOL_COMMAND],
        )

        logger.info("RobotVisualizer initialized")

    def update(self) -> None:
        """
        Update the visualization with the latest
        robot state, joint commands, and tool command.
        """
        # Check for robot state update
        robot_state = self.subscriber.receive(Topic.ROBOT_STATE, timeout=0)

        if robot_state is not None:
            # Extract joint positions from the state
            # The joint_positions is a numpy array already in the correct order
            q = robot_state.joint_positions

            # Update visualization
            self.viz.display(q)
            self.current_q = q

        # Check for joint command update
        joint_command = self.subscriber.receive(Topic.ROBOT_JOINT_COMMAND, timeout=0)

        if joint_command is not None:
            self.viz.display_joint_command(joint_command.joint_positions)

        # Check for tool command update
        tool_command = self.subscriber.receive(Topic.ROBOT_TOOL_COMMAND, timeout=0)

        if tool_command is not None:
            # Visualize the commanded tool pose with the end effector geometry
            self.viz.display_tool_command(tool_command.pose)

    def run(self, rate_hz: float = DEFAULT_RATE_HZ) -> None:
        """Run the visualizer main loop.

        Args:
            rate_hz: Target update rate in Hz (default: 30.0)
        """
        logger.info(f"Starting visualizer main loop at {rate_hz} Hz...")

        try:
            loop_at_rate(self.update, rate_hz=rate_hz)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.close()

    def close(self) -> None:
        """Clean up resources and shut down the visualizer."""
        logger.info("Closing visualizer...")
        self.subscriber.close()
        logger.info("Visualizer closed")


def main():
    visualizer = RobotVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
