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
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotConfig,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)

logger = get_logger(__name__)

# Modes in which the OSC's joint commands are forwarded to the robot. In any
# other mode (HOMING, IDLE) the OSC continuously re-syncs from ROBOT_STATE so
# that reactivation holds the current pose.
OSC_ACTIVE_MODES = {Mode.OCULUS, Mode.KEYBOARD}


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
            topics=[
                Topic.ROBOT_TOOL_COMMAND,
                Topic.ROBOT_BASE_COMMAND,
                Topic.ROBOT_STATE,
                Topic.ORCHESTRATOR_MODE,
            ],
        )
        self.publisher = Publisher()

        # Reference for current tool and base commands
        self.current_tool_command: RobotToolCommand | None = None
        self.current_base_command: RobotBaseCommand | None = None

        # Orchestrator gate. Default to IDLE so we re-sync from state until the
        # orchestrator announces its mode.
        self.current_mode: Mode = Mode.IDLE

    def setup(self) -> None:
        pass

    def _reset_commands_from_state(self, robot_state: RobotState) -> None:
        """Seed current_tool_command/current_base_command from the current robot state.

        Runs FK on the reported joint configuration so the controller holds the
        robot's current pose on reactivation.
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

    @property
    def is_active(self) -> bool:
        return self.current_mode in OSC_ACTIVE_MODES

    def step(self) -> None:
        """Receive tool command and compute joint commands."""
        mode_msg = self.subscriber.receive(Topic.ORCHESTRATOR_MODE)
        if mode_msg is not None:
            if mode_msg.mode != self.current_mode:
                logger.info(f"Orchestrator mode: {self.current_mode} -> {mode_msg.mode}")
            self.current_mode = mode_msg.mode

        robot_state = self.subscriber.receive(Topic.ROBOT_STATE)

        if not self.is_active:
            # Inactive: track the robot continuously so reactivation starts
            # from the current pose. Drain command queues so we don't act on
            # stale messages when reactivated.
            if robot_state is not None:
                self.controller.update_state(robot_state.joint_positions)
                self._reset_commands_from_state(robot_state)
            self.subscriber.receive(Topic.ROBOT_TOOL_COMMAND)
            self.subscriber.receive(Topic.ROBOT_BASE_COMMAND)
            return

        # Active: existing OSC logic.
        # Initialize controller state from robot state feedback until the first
        # command is received, and run open-loop thereafter.
        if robot_state is not None and (
            self.controller.configuration is None or self.current_tool_command is None
        ):
            logger.debug(f"Received robot state: {robot_state}")
            self.controller.update_state(robot_state.joint_positions)

        # Check for new tool command (non-blocking)
        tool_command = self.subscriber.receive(Topic.ROBOT_TOOL_COMMAND)
        if tool_command is not None:
            logger.debug(f"Received tool command: {tool_command}")
            self.current_tool_command = tool_command

        # Check for new base command (non-blocking)
        base_command = self.subscriber.receive(Topic.ROBOT_BASE_COMMAND)
        if base_command is not None:
            logger.debug(f"Received base command: {base_command}")
            self.current_base_command = base_command

        if self.current_tool_command is not None:
            # Compute joint commands using the operational space controller.
            # OSC will merge gripper positions with IK-computed arm positions.
            base_target_pose = (
                self.current_base_command.pose if self.current_base_command is not None else None
            )
            result = self.controller.compute_control(
                self.current_tool_command.pose,
                base_target_pose=base_target_pose,
                gripper_positions=self.current_tool_command.gripper_positions,
            )

            joint_command = RobotJointCommand(
                timestamp=time.perf_counter(), joint_positions=result.q, joint_velocities=result.v
            )

            logger.debug(f"Publishing joint command: {joint_command}")
            self.publisher.publish(joint_command, topic=Topic.CONTROLLER_JOINT_COMMAND)

    def on_close(self) -> None:
        self.subscriber.close()


def main():
    """Main entry point for the robot controller node."""
    controller = RobotController()
    controller.run()


if __name__ == "__main__":
    main()
