import argparse
import math
import time

from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.motors.feetech.controller import POS_MAX, POS_MIN
from humanoid.types.robot import RobotCommand

logger = get_logger(__name__)


def oscillate(
    servo_id: str,
    update_frequency_hz: float = 100,
    period_s: float = 6,
) -> None:
    midpoint = (POS_MAX - POS_MIN) / 2
    angular_freq = 2 * math.pi / period_s

    subscriber = Subscriber(topics=[Topic.ROBOT_STATE])
    state = subscriber.receive(Topic.ROBOT_STATE)
    subscriber.close()

    assert state, "Missing robot state"
    assert servo_id in state.joint_positions, f"Missing joint pos for {servo_id}"

    initial_position = state.joint_positions[servo_id]
    logger.info(f"Current position for servo {servo_id}: {initial_position}")

    normalized_pos = (initial_position - midpoint) / midpoint
    # Clamp to [-1, 1] to handle positions outside normal range
    normalized_pos = max(-1.0, min(1.0, normalized_pos))
    phase_offset = math.asin(normalized_pos)

    start_time = time.time()
    publisher = Publisher()

    def work():
        elapsed = time.time() - start_time
        target = midpoint * math.sin(elapsed * angular_freq + phase_offset) + midpoint
        command = RobotCommand(
            timestamp=time.perf_counter(), joint_positions={servo_id: float(target)}
        )

        logger.debug(f"Publishing command: {command}")
        publisher.publish(command)

    try:
        loop_at_rate(work, update_frequency_hz)
    except KeyboardInterrupt:
        logger.info("Shutting down")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--servo-id",
        type=str,
        required=True,
        help="ID of the servo to oscillate (as string)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=100,
        help="Update frequency in Hz (default: 100)",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=6,
        help="Oscillation period in seconds (default: 6)",
    )
    args = parser.parse_args()

    oscillate(
        servo_id=args.servo_id,
        update_frequency_hz=args.frequency,
        period_s=args.period,
    )


if __name__ == "__main__":
    main()
