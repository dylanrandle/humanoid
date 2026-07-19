import argparse

from humanoid.hardware.actuators.feetech.config import (
    DEFAULT_BAUD_RATE,
    FeetechActuatorControllerConfig,
    FeetechServoType,
)


def add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared Feetech bus-selection arguments to a utility parser."""
    parser.add_argument(
        "--port",
        help="Serial port for the intended controller (default: auto-detect)",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=DEFAULT_BAUD_RATE,
        help=f"Controller baud rate (default: {DEFAULT_BAUD_RATE})",
    )
    parser.add_argument(
        "--servo-type",
        choices=[servo_type.value for servo_type in FeetechServoType],
        default=FeetechServoType.STS.value,
        help=f"Feetech protocol family (default: {FeetechServoType.STS.value})",
    )


def controller_config_from_args(
    args: argparse.Namespace,
) -> FeetechActuatorControllerConfig:
    """Build the controller config selected by shared CLI arguments."""
    return FeetechActuatorControllerConfig(
        port=args.port,
        baud_rate=args.baud_rate,
        servo_type=FeetechServoType(args.servo_type),
    )
