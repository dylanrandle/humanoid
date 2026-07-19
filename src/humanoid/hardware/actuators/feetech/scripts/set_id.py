import argparse

from humanoid.hardware.actuators.feetech.configurator import FeetechActuatorConfigurator
from humanoid.hardware.actuators.feetech.scripts.common import (
    add_connection_arguments,
    controller_config_from_args,
)
from humanoid.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-id", type=int, default=1, help="Current ID of the actuator (default: 1)"
    )
    parser.add_argument(
        "--new-id",
        type=int,
        required=True,
        help="New ID to assign to the actuator (1-253)",
    )
    add_connection_arguments(parser)
    args = parser.parse_args()

    FeetechActuatorConfigurator.set_id(
        args.current_id,
        args.new_id,
        controller_config_from_args(args),
    )


if __name__ == "__main__":
    main()
