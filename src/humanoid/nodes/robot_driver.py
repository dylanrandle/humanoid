import time

import numpy as np

from humanoid.config import IS_SIMULATION, ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.motors.base import MotorController
from humanoid.motors.feetech.controller import FeetechMotorController
from humanoid.motors.simulation import SimulatedMotorController
from humanoid.nodes.base import Node
from humanoid.robots.base import Robot
from humanoid.types.robot import RobotConfig, RobotState
from humanoid.types.servo import ServoControlMode

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 500.0


class RobotDriver(Node):
    def __init__(self, robot_config: RobotConfig = ROBOT_CONFIG, rate_hz: float = DEFAULT_RATE_HZ):
        self.rate_hz = rate_hz
        self.subscriber = Subscriber(topics=[Topic.ROBOT_JOINT_COMMAND])
        self.publisher = Publisher()

        # Use robot configuration from constants
        self.joint_idx_to_servo_id = robot_config.joint_idx_to_servo_id
        self.servo_id_to_joint_idx = robot_config.servo_id_to_joint_idx
        self.sorted_joint_indices = sorted(self.joint_idx_to_servo_id.keys())

        # Load robot model to access joint limits
        self.robot = Robot(robot_config)
        self.joint_lower_limits = self.robot.model.lowerPositionLimit
        self.joint_upper_limits = self.robot.model.upperPositionLimit
        self.joint_velocity_limits = self.robot.model.velocityLimit

        # Store position/velocity controlled joints
        self.position_controlled_joints: list[int] = []
        self.velocity_controlled_joints: list[int] = []
        for joint_idx, servo_id in self.joint_idx_to_servo_id.items():
            control_mode = robot_config.servo_control_modes[servo_id]
            if control_mode == ServoControlMode.POSITION:
                self.position_controlled_joints.append(joint_idx)
            elif control_mode == ServoControlMode.VELOCITY:
                self.velocity_controlled_joints.append(joint_idx)
            else:
                raise ValueError(f"Unrecognized servo control mode: {control_mode}")

        if IS_SIMULATION:
            logger.info("Using simulated motor controller")
            self.controller: MotorController = SimulatedMotorController(robot_config=robot_config)
        else:
            logger.info("Using Feetech motor controller")
            self.controller: MotorController = FeetechMotorController(
                servo_ids=robot_config.servo_ids,
                inverted_servo_ids=robot_config.inverted_servo_ids,
            )

        self.controller.connect()

    def receive(self):
        command = self.subscriber.receive(Topic.ROBOT_JOINT_COMMAND)
        if command is None:
            return

        logger.debug(f"Received command: {command}")

        # Clamp joint positions to respect limits
        clamped_positions = np.clip(
            command.joint_positions, self.joint_lower_limits, self.joint_upper_limits
        )

        positions = {}
        for joint_idx in self.position_controlled_joints:
            servo_id = self.joint_idx_to_servo_id[joint_idx]
            position_idx = self.robot.joint_idx_to_position_idx(joint_idx)
            positions[servo_id] = float(clamped_positions[position_idx])

        velocities = {}
        if command.joint_velocities is not None:
            # Clamp joint velocities to respect limits
            clamped_velocities = np.clip(
                command.joint_velocities, -self.joint_velocity_limits, self.joint_velocity_limits
            )
            for joint_idx in self.velocity_controlled_joints:
                servo_id = self.joint_idx_to_servo_id[joint_idx]
                velocity_idx = self.robot.joint_idx_to_velocity_idx(joint_idx)
                velocities[servo_id] = float(clamped_velocities[velocity_idx])

        self.controller.write_position(positions)
        self.controller.write_velocity(velocities)

    def publish(self):
        positions = self.controller.read_all_positions()
        velocities = self.controller.read_all_velocities()
        temperatures = self.controller.read_all_temperatures()

        joint_idx_to_position = {
            self.servo_id_to_joint_idx[sid]: pos for sid, pos in positions.items()
        }
        joint_idx_to_velocity = {
            self.servo_id_to_joint_idx[sid]: vel for sid, vel in velocities.items()
        }

        motor_temperatures = np.array(
            [
                temperatures.get(self.joint_idx_to_servo_id[joint_idx], 0.0)
                for joint_idx in self.sorted_joint_indices
            ]
        )

        robot_state = RobotState(
            timestamp=time.perf_counter(),
            joint_positions=self.robot.joint_positions_to_q(joint_idx_to_position),
            joint_velocities=self.robot.joint_velocities_to_v(joint_idx_to_velocity),
            motor_temperatures=motor_temperatures,
        )
        logger.debug(f"Measured state: {robot_state}")
        self.publisher.publish(robot_state)

    def setup(self) -> None:
        pass

    def step(self) -> None:
        self.receive()
        self.publish()

    def on_close(self) -> None:
        self.subscriber.close()
        self.controller.disconnect()


def main():
    driver = RobotDriver()
    driver.run()


if __name__ == "__main__":
    main()
