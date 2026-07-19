import argparse

from humanoid.hardware.actuators.feetech.config import FeetechActuatorControllerConfig
from humanoid.hardware.actuators.feetech.configurator import FeetechActuatorConfigurator
from humanoid.hardware.actuators.feetech.scripts.common import (
    add_connection_arguments,
    controller_config_from_args,
)


def read_pid_gains(
    actuator_id: int,
    controller_config: FeetechActuatorControllerConfig | None = None,
) -> dict[str, int]:
    """Read and log PID gains for one Feetech actuator."""
    return FeetechActuatorConfigurator.read_gains(actuator_id, controller_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Feetech actuator PID gains")
    parser.add_argument(
        "--actuator-id",
        type=int,
        required=True,
        help="Actuator ID (1-253)",
    )
    add_connection_arguments(parser)
    args = parser.parse_args()
    read_pid_gains(args.actuator_id, controller_config_from_args(args))


if __name__ == "__main__":
    main()
