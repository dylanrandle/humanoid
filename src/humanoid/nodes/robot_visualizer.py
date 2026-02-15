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
from humanoid.visualizers.meshcat import MeshcatVisualizer

logger = get_logger(__name__)


DEFAULT_RATE_HZ = 30.0


class RobotVisualizer:
    """Visualizer node that displays robot state in real-time.

    This node subscribes to the ROBOT_STATE and ROBOT_TOOL_COMMAND topics
    and updates a MeshCat visualization with the robot's current joint positions
    and commanded tool pose.

    Attributes:
        robot: Robot instance for kinematics and visualization
        viz: MeshCat visualizer instance
        subscriber: LCM subscriber for robot state messages
        running: Flag to control the main loop
    """

    def __init__(self):
        logger.info(f"Initializing robot visualizer for {ROBOT_CONFIG.name}...")
        logger.info(f"Using robot config: {ROBOT_CONFIG}")

        # Load robot model
        self.robot = Robot.from_name(ROBOT_CONFIG.name)
        logger.info(f"Loaded robot: {ROBOT_CONFIG.name}")

        # Setup MeshCat visualization
        logger.info("Setting up MeshCat visualization...")
        self.viz = MeshcatVisualizer(self.robot, show_collisions=VISUALIZER_CONFIG.show_collisions)
        self.viz.initialize(open_browser=VISUALIZER_CONFIG.open_browser)
        logger.info(f"MeshCat visualizer available at: {self.viz.get_url()}")

        # Initialize with home position
        self.current_q = ROBOT_CONFIG.home_position.copy()
        self.viz.display(self.current_q)

        # Setup LCM subscriber
        logger.info("Setting up LCM subscriber...")
        self.subscriber = Subscriber(
            [Topic.ROBOT_STATE, Topic.ROBOT_TOOL_COMMAND],
        )
        logger.info(f"Subscribed to {Topic.ROBOT_STATE.value} and {Topic.ROBOT_TOOL_COMMAND.value}")

        # Track if we've visualized the tool command frame
        self.tool_command_frame_name = "tool_command"

    def update(self) -> None:
        """Update the visualization with the latest robot state and tool command."""
        # Check for robot state update
        robot_state = self.subscriber.receive(Topic.ROBOT_STATE, timeout=0)

        if robot_state is not None:
            # Extract joint positions from the state
            # The joint_positions is a numpy array already in the correct order
            q = robot_state.joint_positions

            # Update visualization
            self.viz.display(q)
            self.current_q = q

        # Check for tool command update
        tool_command = self.subscriber.receive(Topic.ROBOT_TOOL_COMMAND, timeout=0)

        if tool_command is not None:
            # Visualize the commanded tool pose as a coordinate frame
            self.viz.add_frame(
                name=self.tool_command_frame_name,
                pose=tool_command.pose,
            )

    def run(self, rate_hz: float = DEFAULT_RATE_HZ) -> None:
        """Run the visualizer main loop.

        Args:
            rate_hz: Target update rate in Hz (default: 10.0)
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
