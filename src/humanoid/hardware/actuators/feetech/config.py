"""Typed configuration for Feetech actuator hardware."""

from dataclasses import dataclass
from enum import StrEnum

from humanoid.hardware.actuators.config import (
    ActuatorConfig,
    ActuatorControllerConfig,
)

DEFAULT_MAX_ACCELERATION = 254
DEFAULT_BAUD_RATE = 1_000_000
FEETECH_ACTUATOR_ID_MIN = 1
FEETECH_ACTUATOR_ID_MAX = 253
FEETECH_ACCELERATION_MIN = 0
FEETECH_ACCELERATION_MAX = 254


def validate_feetech_acceleration(acceleration: int) -> None:
    """Validate a value before encoding it in the acceleration register."""
    if (
        not isinstance(acceleration, int)
        or not FEETECH_ACCELERATION_MIN <= acceleration <= FEETECH_ACCELERATION_MAX
    ):
        raise ValueError(
            "Feetech acceleration must be between "
            f"{FEETECH_ACCELERATION_MIN} and {FEETECH_ACCELERATION_MAX} "
            "and must be an integer."
        )


class FeetechServoType(StrEnum):
    """Servo protocol supported by the Feetech SDK."""

    STS = "sts"
    HLS = "hls"


@dataclass(frozen=True, kw_only=True)
class FeetechActuatorControllerConfig(ActuatorControllerConfig):
    """Configuration shared by actuators on one Feetech controller."""

    port: str | None = None
    baud_rate: int = DEFAULT_BAUD_RATE
    servo_type: FeetechServoType = FeetechServoType.STS

    def __post_init__(self) -> None:
        if self.port == "":
            raise ValueError("Feetech controller port must not be empty.")
        if self.baud_rate <= 0:
            raise ValueError("Feetech controller baud rate must be positive.")


@dataclass(frozen=True, kw_only=True)
class FeetechActuatorConfig(ActuatorConfig):
    """Configuration for one Feetech actuator."""

    max_acceleration: int = DEFAULT_MAX_ACCELERATION

    def __post_init__(self) -> None:
        if not FEETECH_ACTUATOR_ID_MIN <= self.actuator_id <= FEETECH_ACTUATOR_ID_MAX:
            raise ValueError(
                "Feetech actuator ID must be between "
                f"{FEETECH_ACTUATOR_ID_MIN} and {FEETECH_ACTUATOR_ID_MAX}."
            )
        validate_feetech_acceleration(self.max_acceleration)
