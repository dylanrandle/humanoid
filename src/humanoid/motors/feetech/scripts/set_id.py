import argparse

from humanoid.logging import get_logger
from humanoid.motors.feetech.controller import FeetechConfigurator

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-id", type=int, default=1, help="Current ID of the motor (default: 1)"
    )
    parser.add_argument(
        "--new-id",
        type=int,
        required=True,
        help="New ID to assign to the motor (1-253)",
    )
    args = parser.parse_args()

    FeetechConfigurator.set_id(args.current_id, args.new_id)


if __name__ == "__main__":
    main()
