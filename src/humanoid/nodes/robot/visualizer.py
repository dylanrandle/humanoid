"""MeshCat robot visualization node.

This node subscribes to the ROBOT_STATE LCM channel and visualizes the robot's
current joint positions using MeshCat.
"""

from humanoid.config import ROBOT_CONFIG, VISUALIZER_CONFIG
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.robots.base import Robot
from humanoid.types.robot import RobotConfig
from humanoid.types.visualizer import VisualizerConfig
from humanoid.visualizers.robot import RobotVisualizer as _RobotVisualizer

logger = get_logger(__name__)


class RobotVisualizerNode(Node):
    """Visualizer node that displays robot state in real-time.

    This node subscribes to the ROBOT_STATE, ROBOT_JOINT_COMMAND, and
    ROBOT_TOOL_COMMAND topics and updates a MeshCat visualization with the
    robot's current joint positions, commanded joint positions, and commanded
    tool pose.

    Attributes:
        robot: Robot instance for kinematics and visualization
        viz: MeshCat visualizer instance
        subscriber: LCM subscriber for robot state messages
    """

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        visualizer_config: VisualizerConfig = VISUALIZER_CONFIG,
    ):
        self.rate_hz = 1 / visualizer_config.dt

        # Load robot model
        self.robot = Robot(robot_config)

        # Setup MeshCat visualization
        self.viz_config = visualizer_config
        self.viz = _RobotVisualizer(self.robot, config=visualizer_config)
        self.viz.initialize()

        # Initialize with home position
        self.current_q = robot_config.home_position.copy()
        self.viz.display(self.current_q)

        # Setup LCM subscriber
        self.subscriber = Subscriber(
            [
                Topic.ROBOT_STATE,
                Topic.ROBOT_JOINT_COMMAND,
                Topic.ROBOT_TOOL_COMMAND,
                Topic.ROBOT_BASE_COMMAND,
            ],
        )

    def setup(self) -> None:
        pass

    def step(self) -> None:
        """
        Update the visualization with the latest
        robot state, joint commands, and tool command.
        """
        # Check for robot state update
        robot_state = self.subscriber.receive(Topic.ROBOT_STATE)

        if robot_state is not None:
            # Extract joint positions from the state
            # The joint_positions is a numpy array already in the correct order
            q = robot_state.joint_positions

            # Update visualization
            self.viz.display(q)
            self.current_q = q

        # Check for joint command update
        joint_command = self.subscriber.receive(Topic.ROBOT_JOINT_COMMAND)
        if joint_command is not None and self.viz_config.show_commanded_joint_positions:
            self.viz.display_joint_command(joint_command.joint_positions)

        # Check for tool command update
        tool_command = self.subscriber.receive(Topic.ROBOT_TOOL_COMMAND)
        if tool_command is not None and self.viz_config.show_commanded_tool_pose:
            # Visualize the commanded tool pose with the end effector geometry
            self.viz.display_tool_command(tool_command.pose)

        # Check for base command update
        base_command = self.subscriber.receive(Topic.ROBOT_BASE_COMMAND)
        if base_command is not None and self.viz_config.show_commanded_base_pose:
            self.viz.display_base_command(base_command.pose)

    def on_close(self) -> None:
        self.subscriber.close()


def main():
    visualizer = RobotVisualizerNode()
    visualizer.run()


if __name__ == "__main__":
    main()
