"""Robot node that composes arm, mobile-base, and gripper controllers."""

import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.controllers.base import Controller
from humanoid.controllers.gripper import GripperController
from humanoid.controllers.omniwheel_base import OmniwheelBaseController
from humanoid.controllers.operational_space import OperationalSpaceController
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.robots.base import Robot
from humanoid.types.controllers import ControlResult, OperationalSpaceConfig
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
OSC_ACTIVE_MODES = {Mode.OCULUS, Mode.KEYBOARD, Mode.SYSTEM}


class RobotControllerNode(Node):
    """Node that converts task space commands to joint space commands."""

    def __init__(
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        clock: Callable[[], float] = time.perf_counter,
    ):
        """Initialize the robot controller node.

        Args:
            robot_config: Robot configuration including name and end effector frame
        """
        self.robot = Robot(robot_config)
        self.robot.print_info()

        # Controller strategies each own disjoint model coordinates. Keep the
        # ``controller`` alias while callers migrate to the explicit arm name.
        arm_config = robot_config.operational_space_config or OperationalSpaceConfig()
        self.arm_controller = OperationalSpaceController(robot=self.robot, config=arm_config)
        self.controller = self.arm_controller
        logger.info(f"Initialized arm OSC with config:\n{pformat(asdict(arm_config))}")

        self.base_controller: OmniwheelBaseController | None = None
        base_config = robot_config.omniwheel_base_config
        if base_config is not None:
            self.base_controller = OmniwheelBaseController(robot=self.robot, config=base_config)
            logger.info(
                f"Initialized omniwheel base controller with config:\n"
                f"{pformat(asdict(base_config))}"
            )

        self.gripper_controller: GripperController | None = None
        if robot_config.gripper_joint_indices:
            self.gripper_controller = GripperController(robot=self.robot)
            logger.info("Initialized gripper controller")

        controller_periods = [arm_config.dt]
        if base_config is not None:
            controller_periods.append(base_config.dt)
        if len(set(controller_periods)) != 1:
            raise ValueError("Arm and base controllers must use the same timestep.")
        self.rate_hz = 1 / arm_config.dt
        self._nominal_dt = arm_config.dt
        self._clock = clock
        self._last_control_time: float | None = None

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
            pose=self.robot.get_tool_command_pose(q),
            gripper_positions=gripper_positions,
        )

        base_pose = self.robot.get_base_pose(q)
        if base_pose is not None:
            self.current_base_command = RobotBaseCommand(timestamp=timestamp, pose=base_pose)

    def _update_controller_states(self, joint_positions: NDArray[np.float64]) -> None:
        """Synchronize every controller to one composed full-model state."""
        self.arm_controller.update_state(joint_positions)
        if self.base_controller is not None:
            self.base_controller.update_state(joint_positions)
        if self.gripper_controller is not None:
            self.gripper_controller.update_state(joint_positions)

    def _apply_gripper_target_to_model_state(self, dt: float) -> ControlResult | None:
        """Apply the latest gripper command before model-based controllers solve."""
        gripper_positions = (
            self.current_tool_command.gripper_positions
            if self.current_tool_command is not None
            else None
        )
        if self.gripper_controller is None or gripper_positions is None:
            return None

        result = self.gripper_controller.compute_control(gripper_positions, dt=dt)
        self.arm_controller.update_state(result.q)
        if self.base_controller is not None:
            self.base_controller.update_state(result.q)
        return result

    @property
    def is_active(self) -> bool:
        return self.current_mode in OSC_ACTIVE_MODES

    def step(self) -> None:
        """Receive tool command and compute joint commands."""
        mode_msg = self.subscriber.receive(Topic.ORCHESTRATOR_MODE)
        if mode_msg is not None:
            if mode_msg.mode != self.current_mode:
                logger.debug(f"Updated mode: {self.current_mode} -> {mode_msg.mode}")
                self._last_control_time = None
                self.arm_controller.reset_motion()
            self.current_mode = mode_msg.mode

        robot_state = self.subscriber.receive(Topic.ROBOT_STATE)

        if not self.is_active:
            # Inactive: track the robot continuously so reactivation starts
            # from the current pose. Drain command queues so we don't act on
            # stale messages when reactivated.
            if robot_state is not None:
                self._update_controller_states(robot_state.joint_positions)
                self._reset_commands_from_state(robot_state)
            self.subscriber.receive(Topic.ROBOT_TOOL_COMMAND)
            self.subscriber.receive(Topic.ROBOT_BASE_COMMAND)
            return

        # Active: existing OSC logic.
        # Initialize controller state from robot state feedback until the first
        # command is received, and run open-loop thereafter.
        if robot_state is not None and (
            self.arm_controller.configuration is None
            or (self.current_tool_command is None and self.current_base_command is None)
        ):
            logger.debug(f"Received robot state: {robot_state}")
            self._update_controller_states(robot_state.joint_positions)

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

        dt = self._control_timestep()

        # Apply the latest gripper target to the shared model state before solving
        # arm IK. The OSC collision model therefore sees the commanded finger pose
        # from this tick rather than the previous command.
        gripper_result = self._apply_gripper_target_to_model_state(dt)

        results: list[tuple[Controller[pin.SE3], ControlResult]] = []
        if self.current_tool_command is not None:
            results.append(
                (
                    self.arm_controller,
                    self.arm_controller.compute_control(self.current_tool_command.pose, dt=dt),
                )
            )
        if self.base_controller is not None and self.current_base_command is not None:
            results.append(
                (
                    self.base_controller,
                    self.base_controller.compute_control(self.current_base_command.pose, dt=dt),
                )
            )
        if not results:
            return

        # Each result contains the full model for compatibility with the driver,
        # but only the controller-owned coordinates are copied into the command.
        q = results[0][1].q.copy()
        v = np.zeros(self.robot.model.nv)
        for controller, result in results:
            q[controller.controlled_q_indices] = result.q[controller.controlled_q_indices]
            v[controller.controlled_v_indices] = result.v[controller.controlled_v_indices]

        if self.gripper_controller is not None and gripper_result is not None:
            q[self.gripper_controller.controlled_q_indices] = gripper_result.q[
                self.gripper_controller.controlled_q_indices
            ]
            v[self.gripper_controller.controlled_v_indices] = gripper_result.v[
                self.gripper_controller.controlled_v_indices
            ]

        # Keep each strategy's non-owned coordinates current for FK and collision
        # calculations on the next tick.
        self._update_controller_states(q)

        joint_command = RobotJointCommand(
            timestamp=time.perf_counter(), joint_positions=q, joint_velocities=v
        )

        logger.debug(f"Publishing joint command: {joint_command}")
        self.publisher.publish(joint_command, topic=Topic.CONTROLLER_JOINT_COMMAND)

    def _control_timestep(self) -> float:
        """Return measured elapsed time, bounded around the configured period."""
        now = self._clock()
        if self._last_control_time is None:
            dt = self._nominal_dt
        else:
            elapsed = now - self._last_control_time
            dt = float(np.clip(elapsed, self._nominal_dt * 0.5, self._nominal_dt * 2.0))
        self._last_control_time = now
        return dt

    def on_close(self) -> None:
        self.subscriber.close()


def main():
    """Main entry point for the robot controller node."""
    controller = RobotControllerNode()
    controller.run()


if __name__ == "__main__":
    main()
