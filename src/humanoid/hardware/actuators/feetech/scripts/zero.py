import argparse

from humanoid.hardware.actuators.feetech.configurator import FeetechActuatorConfigurator
from humanoid.hardware.actuators.feetech.scripts.common import (
    add_connection_arguments,
    controller_config_from_args,
)
from humanoid.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the zero position for a Feetech actuator")
    parser.add_argument(
        "--actuator-id",
        type=int,
        required=True,
        help="ID of the actuator to set zero position (1-253)",
    )
    add_connection_arguments(parser)
    args = parser.parse_args()

    FeetechActuatorConfigurator.set_zero(
        args.actuator_id,
        controller_config_from_args(args),
    )


if __name__ == "__main__":
    main()
