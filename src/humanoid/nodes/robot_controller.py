import time

import numpy as np

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.controllers.operational_space import OperationalSpaceController
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.robots.base import Robot
from humanoid.types.robot import RobotConfig, RobotJointCommand

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 500.0


class RobotController:
    """Node that converts task space commands to joint space commands."""

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        """Initialize the robot controller node.

        Args:
            rate_hz: Control loop rate in Hz
        """
        logger.info(f"Initializing RobotController for: {robot_config.name}")

        self.rate_hz = rate_hz
        self.robot = Robot.from_name(robot_config.name)
        self.robot.print_info()

        # Initialize operational space controller
        logger.info(f"Initializing OSC for frame: {robot_config.end_effector_frame}")
        self.controller = OperationalSpaceController(
            robot=self.robot,
            end_effector_frame=robot_config.end_effector_frame,
        )

        # Set up LCM communication
        self.subscriber = Subscriber(
            topics=[Topic.ROBOT_TOOL_COMMAND, Topic.ROBOT_STATE],
        )
        self.publisher = Publisher()

        # Initialize state from robot config home position
        self.q_current = robot_config.home_position.copy()
        self.v_current = np.zeros(self.controller.nv)

        # Set joint centers for null space control
        self.controller.set_joint_centers(self.q_current)

        logger.info("RobotController initialized")

    def receive_and_compute(self) -> None:
        """Receive tool command and compute joint commands."""
        # Check for robot state update
        robot_state = self.subscriber.receive(Topic.ROBOT_STATE, timeout=0)

        # Update internal state from robot state feedback
        if robot_state is not None:
            # Use joint positions and velocities directly from robot state
            # The driver already handles velocity estimation with filtering
            self.q_current = robot_state.joint_positions.copy()
            self.v_current = robot_state.joint_velocities.copy()
            logger.debug(
                f"Updated state from robot feedback: q={self.q_current}, v={self.v_current}"
            )

        # Check for new tool command (non-blocking)
        tool_command = self.subscriber.receive(Topic.ROBOT_TOOL_COMMAND, timeout=0)

        if tool_command is not None:
            logger.debug(f"Received tool command: position={tool_command.pose.translation}")

            # Compute joint commands using the operational space controller
            q_cmd = self.controller.compute_joint_commands(
                q_current=self.q_current,
                v_current=self.v_current,
                target_pose=tool_command.pose,
                target_velocity=None,  # Let the controller compute velocity
            )

            # Create joint command message with np.ndarray
            joint_command = RobotJointCommand(
                timestamp=time.perf_counter(),
                joint_positions=q_cmd,
            )

            # Publish joint command
            logger.debug(f"Publishing joint command: {joint_command}")
            self.publisher.publish(joint_command)

    def run(self) -> None:
        """Run the controller node main loop.

        Args:
            rate_hz: Control loop rate in Hz
        """
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
