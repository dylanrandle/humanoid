import time

from humanoid.constants import SERVO_IDS, Topic
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.motors.feetech.controller import FeetechController
from humanoid.types.robot import RobotState

logger = get_logger(__name__)

DEFAULT_RATE_HZ = 100.0


class RobotDriver:
    def __init__(self):
        self.subscriber = Subscriber(topics=[Topic.ROBOT_COMMAND])
        self.publisher = Publisher()
        self.controller = FeetechController(servo_ids=SERVO_IDS)
        self.controller.connect()
        logger.info("Initialized")

    def receive(self):
        command = self.subscriber.receive(Topic.ROBOT_COMMAND, timeout=0)
        if command is not None:
            logger.debug(f"Received command: {command}")
            positions = {int(k): v for k, v in command.joint_positions.items()}
            self.controller.write_position(positions)

    def publish(self):
        positions = self.controller.read_all_positions()
        temperatures = self.controller.read_all_temperatures()
        robot_state = RobotState(
            timestamp=time.perf_counter(),
            joint_positions={str(k): v for k, v in positions.items()},
            motor_temperatures={str(k): v for k, v in temperatures.items()},
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
