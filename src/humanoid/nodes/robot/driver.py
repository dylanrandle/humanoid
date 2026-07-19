"""Robot hardware driver node and injectable actuator-test harness."""

import time
from collections.abc import Callable

import numpy as np
import pinocchio as pin

from humanoid.config import ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.hardware.actuators.factory import create_actuator_system
from humanoid.hardware.actuators.system import ActuatorSystem
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.robots.base import Robot
from humanoid.robots.command import normalize_robot_joint_command
from humanoid.robots.watchdog import VelocityCommandWatchdog
from humanoid.state_estimation.root.base import (
    RootState,
    RootStateEstimator,
)
from humanoid.state_estimation.root.factory import create_root_state_estimator
from humanoid.types.actuator import ActuatorControlMode
from humanoid.types.robot import RobotConfig, RobotState

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 500.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 0.25


class RobotDriverNode(Node):
    def __init__(  # noqa: PLR0913 - hardware dependencies are intentionally injectable
        self,
        robot_config: RobotConfig = ROBOT_CONFIG,
        rate_hz: float = DEFAULT_RATE_HZ,
        actuator_system: ActuatorSystem | None = None,
        root_state_estimator: RootStateEstimator | None = None,
        command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.rate_hz = rate_hz
        self.subscriber = Subscriber(topics=[Topic.ROBOT_JOINT_COMMAND])
        self.publisher = Publisher()

        self.robot = Robot(robot_config)
        self.actuator_control_modes = robot_config.actuator_control_modes
        self.actuator_joint_names = self.robot.actuator_joint_names
        self.joint_indices = {
            joint_name: self.robot.joint_name_to_idx(joint_name)
            for joint_name in self.actuator_joint_names
        }

        self.joint_lower_limits = self.robot.model.lowerPositionLimit
        self.joint_upper_limits = self.robot.model.upperPositionLimit
        self.joint_velocity_limits = self.robot.model.velocityLimit

        self.position_controlled_joints = [
            joint_name
            for joint_name in self.actuator_joint_names
            if self.actuator_control_modes[joint_name] is ActuatorControlMode.POSITION
        ]
        self.velocity_controlled_joints = [
            joint_name
            for joint_name in self.actuator_joint_names
            if self.actuator_control_modes[joint_name] is ActuatorControlMode.VELOCITY
        ]

        self.actuator_system = actuator_system or create_actuator_system(
            self.actuator_control_modes,
            robot_config.hardware.actuators if robot_config.hardware is not None else None,
        )
        self._command_watchdog = VelocityCommandWatchdog(
            self._stop_actuators,
            command_timeout_seconds,
            clock,
        )
        self._root_q_slice = self.robot.get_root_q_slice()
        self._root_v_slice = self.robot.get_root_v_slice()
        self.root_state_estimator: RootStateEstimator | None = None
        if self._root_q_slice is not None and self._root_v_slice is not None:
            initial_root_state = RootState(
                position=pin.neutral(self.robot.model)[self._root_q_slice].copy(),
                velocity=np.zeros(self.robot.model.nv)[self._root_v_slice],
            )
            state_estimation = robot_config.state_estimation
            if state_estimation is None or state_estimation.root is None:
                raise RuntimeError("Mobile robot requires root-state estimator config.")
            self.root_state_estimator = root_state_estimator or create_root_state_estimator(
                state_estimation.root,
                self.robot,
                initial_root_state,
            )

        self.actuator_system.connect()

    def receive(self) -> None:
        command = self.subscriber.receive(Topic.ROBOT_JOINT_COMMAND)
        if command is None:
            if self._command_watchdog.stop_if_stale():
                logger.warning("Robot command watchdog stopped velocity-controlled actuators")
            return

        logger.debug("Received command: %s", command)
        normalized = normalize_robot_joint_command(
            command,
            self.joint_lower_limits,
            self.joint_upper_limits,
            self.joint_velocity_limits,
        )
        positions = {
            joint_name: float(
                normalized.joint_positions[
                    self.robot.joint_idx_to_position_idx(self.joint_indices[joint_name])
                ]
            )
            for joint_name in self.position_controlled_joints
        }

        velocities = {
            joint_name: float(
                normalized.joint_velocities[
                    self.robot.joint_idx_to_velocity_idx(self.joint_indices[joint_name])
                ]
            )
            for joint_name in self.velocity_controlled_joints
        }

        self.actuator_system.write_positions(positions)
        self.actuator_system.write_velocities(velocities)
        self._command_watchdog.observe_command(velocity_active=any(velocities.values()))

    def _stop_actuators(self) -> None:
        self.actuator_system.stop()

    def publish(self) -> None:
        actuator_states = self.actuator_system.read_states()
        missing_positions = [
            joint_name
            for joint_name in self.actuator_joint_names
            if joint_name not in actuator_states or actuator_states[joint_name].position is None
        ]
        missing_velocities = [
            joint_name
            for joint_name in self.actuator_joint_names
            if joint_name not in actuator_states or actuator_states[joint_name].velocity is None
        ]
        if missing_positions or missing_velocities:
            details = []
            if missing_positions:
                details.append(f"positions: {', '.join(missing_positions)}")
            if missing_velocities:
                details.append(f"velocities: {', '.join(missing_velocities)}")
            raise RuntimeError(f"Incomplete actuator feedback ({'; '.join(details)}).")
        joint_idx_to_position = {
            self.joint_indices[joint_name]: state.position
            for joint_name, state in actuator_states.items()
            if state.position is not None
        }
        joint_idx_to_velocity = {
            self.joint_indices[joint_name]: state.velocity
            for joint_name, state in actuator_states.items()
            if state.velocity is not None
        }
        actuator_temperatures = np.array(
            [
                (
                    actuator_states[joint_name].temperature
                    if joint_name in actuator_states
                    and actuator_states[joint_name].temperature is not None
                    else 0.0
                )
                for joint_name in self.actuator_joint_names
            ]
        )

        q = self.robot.joint_positions_to_q(joint_idx_to_position)
        v = self.robot.joint_velocities_to_v(joint_idx_to_velocity)
        if (
            self.root_state_estimator is not None
            and self._root_q_slice is not None
            and self._root_v_slice is not None
        ):
            root_state = self.root_state_estimator.update(q, v)
            q[self._root_q_slice] = root_state.position
            v[self._root_v_slice] = root_state.velocity

        robot_state = RobotState(
            timestamp=time.perf_counter(),
            joint_positions=q,
            joint_velocities=v,
            actuator_temperatures=actuator_temperatures,
        )
        logger.debug("Measured state: %s", robot_state)
        self.publisher.publish(robot_state, topic=Topic.ROBOT_STATE)

    def setup(self) -> None:
        pass

    def step(self) -> None:
        self.receive()
        self.publish()

    def on_close(self) -> None:
        try:
            self._command_watchdog.stop()
        finally:
            try:
                self.subscriber.close()
            finally:
                self.actuator_system.disconnect()


def main() -> None:
    driver = RobotDriverNode()
    driver.run()


if __name__ == "__main__":
    main()
