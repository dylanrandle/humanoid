import time
from dataclasses import asdict
from pprint import pformat

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.controllers.operational_space import (
    OperationalSpaceConfig,
    OperationalSpaceController,
)
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.robots.base import Robot
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotConfig,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

logger = get_logger(__name__)


class RobotController(Node):
    """Node that converts task space commands to joint space commands."""

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
    ):
        """Initialize the robot controller node.

        Args:
            robot_config: Robot configuration including name and end effector frame
            rate_hz: Control loop rate in Hz
        """
        self.robot = Robot(robot_config)
        self.robot.print_info()

        # Initialize operational space controller
        config = robot_config.operational_space_config or OperationalSpaceConfig()
        self.controller = OperationalSpaceController(robot=self.robot, config=config)
        logger.info(f"Initialized OSC with config:\n{pformat(asdict(config))}")

        self.rate_hz = 1 / config.dt

        # Set up LCM communication
        self.subscriber = Subscriber(
            topics=[Topic.ROBOT_TOOL_COMMAND, Topic.ROBOT_BASE_COMMAND, Topic.ROBOT_STATE],
        )
        self.publisher = Publisher()

        # Reference for current tool and base commands
        self.current_tool_command: RobotToolCommand | None = None
        self.current_base_command: RobotBaseCommand | None = None

    def setup(self) -> None:
        pass

    def _initialize_commands_from_state(self, robot_state: RobotState) -> None:
        """Seed current_tool_command/current_base_command from the current robot state.

        Runs FK on the reported joint configuration so the controller holds the
        robot's current pose until an external command arrives.
        """
        q = robot_state.joint_positions
        timestamp = time.perf_counter()

        gripper_indices = self.robot.get_gripper_position_indices()
        gripper_positions = q[gripper_indices] if gripper_indices else None

        self.current_tool_command = RobotToolCommand(
            timestamp=timestamp,
            pose=self.robot.get_tool_pose(q),
            gripper_positions=gripper_positions,
        )

        base_pose = self.robot.get_base_pose(q)
        if base_pose is not None:
            self.current_base_command = RobotBaseCommand(timestamp=timestamp, pose=base_pose)

    def step(self) -> None:
        """Receive tool command and compute joint commands."""
        # Check for robot state update (for initialization only)
        robot_state = self.subscriber.receive(Topic.ROBOT_STATE)

        # Initialize controller state from robot state feedback until
        # the first command is received, and run open-loop thereafter
        if robot_state is not None and (
            self.controller.configuration is None or self.current_tool_command is None
        ):
            logger.debug(f"Received robot state: {robot_state}")
            self.controller.update_state(robot_state.joint_positions)
            # TODO: enable once we handle policy multiplexing
            # self._initialize_commands_from_state(robot_state)

        # Check for new tool command (non-blocking)
        tool_command = self.subscriber.receive(Topic.ROBOT_TOOL_COMMAND)

        # Update internal state from tool command
        if tool_command is not None:
            logger.debug(f"Received tool command: {tool_command}")
            self.current_tool_command = tool_command

        # Check for new base command (non-blocking)
        base_command = self.subscriber.receive(Topic.ROBOT_BASE_COMMAND)

        if base_command is not None:
            logger.debug(f"Received base command: {base_command}")
            self.current_base_command = base_command

        # Control current command
        if self.current_tool_command is not None:
            # Compute joint commands using the operational space controller
            # OSC will merge gripper positions with IK-computed arm positions
            base_target_pose = (
                self.current_base_command.pose if self.current_base_command is not None else None
            )
            result = self.controller.compute_control(
                self.current_tool_command.pose,
                base_target_pose=base_target_pose,
                gripper_positions=self.current_tool_command.gripper_positions,
            )

            # Create joint command message with np.ndarray
            joint_command = RobotJointCommand(
                timestamp=time.perf_counter(), joint_positions=result.q, joint_velocities=result.v
            )

            # Publish joint command
            logger.debug(f"Publishing joint command: {joint_command}")
            self.publisher.publish(joint_command)

    def on_close(self) -> None:
        self.subscriber.close()


def main():
    """Main entry point for the robot controller node."""
    controller = RobotController()
    controller.run()


if __name__ == "__main__":
    main()
