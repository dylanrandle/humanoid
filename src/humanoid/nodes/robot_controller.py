import time

import numpy as np

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.controllers.operational_space import OperationalSpaceController
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.robots.base import Robot
from humanoid.types.robot import RobotJointCommand

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class RobotController:
    """Node that converts task space commands to joint space commands."""

    def __init__(
        self,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        """Initialize the robot controller node.

        Args:
            rate_hz: Control loop rate in Hz
        """
        logger.info(f"Initializing RobotController for robot: {ROBOT_CONFIG.name}")
        logger.info(f"Using robot config: {ROBOT_CONFIG}")

        self.rate_hz = rate_hz

        # Load robot model
        logger.info("Loading robot model...")
        self.robot = Robot.from_name(ROBOT_CONFIG.name)

        # Initialize operational space controller
        logger.info(f"Initializing OSC for frame: {ROBOT_CONFIG.end_effector_frame}")
        self.controller = OperationalSpaceController(
            robot=self.robot,
            end_effector_frame=ROBOT_CONFIG.end_effector_frame,
        )

        # Set up LCM communication
        logger.info("Setting up LCM communication...")
        self.subscriber = Subscriber(topics=[Topic.ROBOT_TOOL_COMMAND])
        self.publisher = Publisher()

        # Initialize state from robot config home position
        self.q_current = ROBOT_CONFIG.home_position.copy()
        self.v_current = np.zeros(self.controller.nv)

        # Set joint centers for null space control
        self.controller.set_joint_centers(self.q_current)

        logger.info("Initialization complete")

    def receive_and_compute(self) -> None:
        """Receive tool command and compute joint commands."""
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

            # Update current state (in a real system, this would come from robot state feedback)
            # For now, we assume perfect tracking
            self.q_current = q_cmd.copy()
            self.v_current = (q_cmd - self.q_current) / self.controller.config.dt

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

        def work():
            self.receive_and_compute()

        try:
            loop_at_rate(work, rate_hz=self.rate_hz)
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
