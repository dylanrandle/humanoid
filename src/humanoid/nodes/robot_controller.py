import time

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.controllers.operational_space import (
    OperationalSpaceConfig,
    OperationalSpaceController,
)
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.robots.base import Robot
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotConfig,
    RobotJointCommand,
    RobotToolCommand,
)

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 200.0


class RobotController:
    """Node that converts task space commands to joint space commands."""

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        """Initialize the robot controller node.

        Args:
            robot_config: Robot configuration including name and end effector frame
            rate_hz: Control loop rate in Hz
        """
        logger.info(f"Initializing RobotController for: {robot_config.name}")

        self.rate_hz = rate_hz
        self.dt = 1 / rate_hz
        self.robot = Robot(robot_config)
        self.robot.print_info()

        # Initialize operational space controller
        logger.info(f"Initializing OSC for frame: {robot_config.tool_frame}")
        config = OperationalSpaceConfig(dt=self.dt)
        self.controller = OperationalSpaceController(robot=self.robot, config=config)

        # Set up LCM communication
        self.subscriber = Subscriber(
            topics=[Topic.ROBOT_TOOL_COMMAND, Topic.ROBOT_BASE_COMMAND, Topic.ROBOT_STATE],
        )
        self.publisher = Publisher()

        # Reference for current tool and base commands
        self.current_tool_command: RobotToolCommand | None = None
        self.current_base_command: RobotBaseCommand | None = None

        logger.info("RobotController initialized")

    # TODO: initial tool/base commands with robot state on first update as well
    def receive_and_compute(self) -> None:
        """Receive tool command and compute joint commands."""
        # Check for robot state update (for initialization only)
        robot_state = self.subscriber.receive(Topic.ROBOT_STATE)

        # Initialize controller state from first robot state feedback
        # After initialization, use open-loop control (internal state integration)
        if robot_state is not None and self.controller.configuration is None:
            logger.info(f"Initializing controller state from robot state: {robot_state}")
            self.controller.update_state(robot_state.joint_positions)

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

    def run(self) -> None:
        """Run the controller node main loop."""
        logger.info(f"Starting controller loop at {self.rate_hz} Hz...")

        try:
            loop_at_rate(self.receive_and_compute, rate_hz=self.rate_hz)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.close()

    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing RobotController...")
        self.subscriber.close()
        logger.info("RobotController closed")


def main():
    """Main entry point for the robot controller node."""
    controller = RobotController()
    controller.run()


if __name__ == "__main__":
    main()
