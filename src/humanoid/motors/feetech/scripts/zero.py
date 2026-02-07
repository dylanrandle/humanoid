import argparse

from humanoid.logging import get_logger
from humanoid.motors.feetech.controller import FeetechConfigurator

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Set the zero position for a Feetech motor")
    parser.add_argument(
        "--motor-id",
        type=int,
        required=True,
        help="ID of the motor to set zero position (1-253)",
    )
    args = parser.parse_args()

    FeetechConfigurator.set_zero(args.motor_id)


if __name__ == "__main__":
    main()
