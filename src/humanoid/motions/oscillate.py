import argparse
import math
import time

from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.types.robot import RobotCommand

logger = get_logger(__name__)


DEFAULT_RATE_HZ = 50


def oscillate(
    servo_id: str,
    update_frequency_hz: float = DEFAULT_RATE_HZ,
    period_s: float = 5,
) -> None:
    subscriber = Subscriber(topics=[Topic.ROBOT_STATE])
    state = subscriber.receive(Topic.ROBOT_STATE)
    subscriber.close()

    assert state, "Missing robot state"
    assert servo_id in state.joint_positions, f"Missing joint pos for {servo_id}"

    initial_angle = state.joint_positions[servo_id]
    # Normalize angle to [-1, 1] range (assuming max range is -π to π)
    normalized_angle = initial_angle / math.pi
    # Clamp to [-1, 1] to handle angles outside normal range
    normalized_angle = max(-1.0, min(1.0, normalized_angle))
    phase_offset = math.asin(normalized_angle)

    angular_freq = 2 * math.pi / period_s
    start_time = time.time()
    publisher = Publisher()

    def work():
        elapsed = time.time() - start_time
        # Oscillate between -π and π, centered at 0
        target_angle = math.pi * math.sin(elapsed * angular_freq + phase_offset)
        command = RobotCommand(
            timestamp=time.perf_counter(), joint_positions={servo_id: float(target_angle)}
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
        default=DEFAULT_RATE_HZ,
        help=f"Update frequency in Hz (default: {DEFAULT_RATE_HZ})",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=5,
        help="Oscillation period in seconds (default: 5)",
    )
    args = parser.parse_args()

    oscillate(
        servo_id=args.servo_id,
        update_frequency_hz=args.frequency,
        period_s=args.period,
    )


if __name__ == "__main__":
    main()
