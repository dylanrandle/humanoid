import argparse

from humanoid.hardware.actuators.feetech.configurator import FeetechActuatorConfigurator
from humanoid.hardware.actuators.feetech.scripts.common import (
    add_connection_arguments,
    controller_config_from_args,
)
from humanoid.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a Feetech controller bus")
    add_connection_arguments(parser)
    args = parser.parse_args()
    logger.info("Scanning for Feetech actuators")
    results = FeetechActuatorConfigurator.scan(controller_config_from_args(args))
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
