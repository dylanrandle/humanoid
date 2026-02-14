import argparse
import math
import time

from humanoid.constants import ROBOT_CONFIG, Topic
from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate
from humanoid.middleware.lcm import Publisher, Subscriber
from humanoid.types.robot import RobotJointCommand

logger = get_logger(__name__)


DEFAULT_RATE_HZ = 50


def oscillate(
    servo_ids: list[str] | None = None,
    update_frequency_hz: float = DEFAULT_RATE_HZ,
    period_s: float = 5,
) -> None:
    # Default to all servos if none specified
    if servo_ids is None:
        servo_ids = [str(sid) for sid in ROBOT_CONFIG.servo_ids]
        logger.info(f"Oscillating all servos for {ROBOT_CONFIG.name}: {servo_ids}")
    else:
        logger.info(f"Oscillating servos: {servo_ids}")

    # Get initial state for all servos
    subscriber = Subscriber(topics=[Topic.ROBOT_STATE])
    state = subscriber.receive(Topic.ROBOT_STATE)
    subscriber.close()

    assert state, "Missing robot state"

    # Calculate initial phase offsets for each servo
    servo_data = {}
    for servo_id in servo_ids:
        assert servo_id in state.joint_positions, f"Missing joint pos for {servo_id}"

        initial_angle = state.joint_positions[servo_id]
        # Normalize angle to [-1, 1] range (assuming max range is -π to π)
        normalized_angle = initial_angle / math.pi
        # Clamp to [-1, 1] to handle angles outside normal range
        normalized_angle = max(-1.0, min(1.0, normalized_angle))
        phase_offset = math.asin(normalized_angle)

        servo_data[servo_id] = {
            "phase_offset": phase_offset,
        }

    angular_freq = 2 * math.pi / period_s
    start_time = time.time()
    publisher = Publisher()

    def work():
        elapsed = time.time() - start_time
        joint_positions = {}

        # Calculate target angle for each servo
        for servo_id in servo_ids:
            phase_offset = servo_data[servo_id]["phase_offset"]
            # Oscillate between -π and π, centered at 0
            target_angle = math.pi * math.sin(elapsed * angular_freq + phase_offset)
            joint_positions[servo_id] = float(target_angle)

        command = RobotJointCommand(
            timestamp=time.perf_counter(),
            joint_positions=joint_positions,
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
        "--servo-ids",
        type=str,
        nargs="+",
        help="IDs of the servos to oscillate. If not specified, oscillates all servos.",
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
        servo_ids=args.servo_ids,
        update_frequency_hz=args.frequency,
        period_s=args.period,
    )


if __name__ == "__main__":
    main()
