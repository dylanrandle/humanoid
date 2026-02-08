import argparse

from humanoid.logger import get_logger
from humanoid.motors.feetech.configurator import FeetechConfigurator

logger = get_logger(__name__)


def set_pid_gains(servo_id: int, p_gain: int, i_gain: int, d_gain: int) -> bool:
    logger.info(f"Setting PID gains for motor {servo_id}: P={p_gain}, I={i_gain}, D={d_gain}")

    FeetechConfigurator.read_gains(servo_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="Motor ID (1-253)",
    )
    parser.add_argument(
        "--p",
        type=int,
        required=True,
        help="Proportional gain",
    )
    parser.add_argument(
        "--i",
        type=int,
        required=True,
        help="Integral gain",
    )
    parser.add_argument(
        "--d",
        type=int,
        required=True,
        help="Derivative gain",
    )
    args = parser.parse_args()

    set_pid_gains(args.id, args.p, args.i, args.d)


if __name__ == "__main__":
    main()
