import time

import numpy as np

from humanoid.config import IS_SIMULATION, ROBOT_CONFIG
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.motors.base import MotorController
from humanoid.motors.feetech.controller import FeetechMotorController
from humanoid.motors.simulation import SimulatedMotorController
from humanoid.types.robot import RobotState

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class RobotDriver:
    def __init__(self):
        self.subscriber = Subscriber(topics=[Topic.ROBOT_JOINT_COMMAND])
        self.publisher = Publisher()

        # Use robot configuration from constants
        logger.info(f"Using robot config: {ROBOT_CONFIG}")
        servo_ids = ROBOT_CONFIG.servo_ids
        self.joint_idx_to_servo_id = ROBOT_CONFIG.joint_idx_to_servo_id
        # Create reverse mapping from servo ID to joint index
        self.servo_id_to_joint_idx = {v: k for k, v in self.joint_idx_to_servo_id.items()}

        if IS_SIMULATION:
            logger.info("Using simulated motor controller")
            self.controller: MotorController = SimulatedMotorController(servo_ids=servo_ids)
        else:
            logger.info("Using Feetech motor controller")
            self.controller: MotorController = FeetechMotorController(servo_ids=servo_ids)

        self.controller.connect()
        logger.info("Initialized")

    def receive(self):
        command = self.subscriber.receive(Topic.ROBOT_JOINT_COMMAND, timeout=0)
        if command is not None:
            logger.debug(f"Received command: {command}")
            # Convert joint indices to servo IDs using the mapping
            positions = {
                self.joint_idx_to_servo_id[joint_idx]: float(position)
                for joint_idx, position in enumerate(command.joint_positions)
            }
            self.controller.write_position(positions)

    def publish(self):
        positions = self.controller.read_all_positions()
        temperatures = self.controller.read_all_temperatures()

        # Convert servo positions to joint position array
        joint_positions = np.zeros(len(self.joint_idx_to_servo_id))
        for servo_id, position in positions.items():
            joint_idx = self.servo_id_to_joint_idx[servo_id]
            joint_positions[joint_idx] = position

        # Convert servo temperatures to motor temperature array
        motor_temperatures = np.zeros(len(self.joint_idx_to_servo_id))
        for servo_id, temperature in temperatures.items():
            joint_idx = self.servo_id_to_joint_idx[servo_id]
            motor_temperatures[joint_idx] = temperature

        robot_state = RobotState(
            timestamp=time.perf_counter(),
            joint_positions=joint_positions,
            motor_temperatures=motor_temperatures,
        )
        logger.debug(f"Measured state: {robot_state}")
        self.publisher.publish(robot_state)

    def run(self, rate_hz: float = DEFAULT_RATE_HZ) -> None:
        logger.info(f"Starting main loop at {rate_hz} Hz...")

        def work():
            self.receive()
            self.publish()

        try:
            loop_at_rate(work, rate_hz=rate_hz)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.close()

    def close(self) -> None:
        logger.info("Closing...")
        self.subscriber.close()
        self.controller.disconnect()


def main():
    driver = RobotDriver()
    driver.run()


if __name__ == "__main__":
    main()
