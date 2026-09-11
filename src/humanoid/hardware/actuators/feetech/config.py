"""Typed configuration for Feetech actuator hardware."""

import math
from dataclasses import dataclass
from enum import StrEnum

from humanoid.types.actuator import (
    ActuatorConfig,
    ActuatorControllerConfig,
)

# The register uses 8.7 degrees/s^2 per unit. A conservative value of 15
# corresponds to roughly 2.3 rad/s^2 and matches the maintenance jog utility.
DEFAULT_MAX_ACCELERATION = 15
DEFAULT_MAX_POSITION_VELOCITY = 1.0
# Additional position-mode speed budget in rad/s per radian of tracking error.
DEFAULT_POSITION_TRACKING_ERROR_GAIN = 10.0
DEFAULT_BAUD_RATE = 1_000_000
FEETECH_ACTUATOR_ID_MIN = 1
FEETECH_ACTUATOR_ID_MAX = 253
FEETECH_ACCELERATION_MIN = 0
FEETECH_ACCELERATION_MAX = 254
FEETECH_GAIN_MIN = 0
FEETECH_GAIN_MAX = 254
# Feetech documents the acceleration register in 8.7 degree/s^2 increments.
FEETECH_ACCELERATION_UNIT_RAD_S2 = math.radians(8.7)
# 0.732 RPM per raw speed unit.
FEETECH_SPEED_UNIT_RAD_S = 0.732 * 2 * math.pi / 60


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


def _validate_gain(name: str, value: int) -> None:
    if not isinstance(value, int) or not FEETECH_GAIN_MIN <= value <= FEETECH_GAIN_MAX:
        raise ValueError(
            f"Feetech {name} gain must be between {FEETECH_GAIN_MIN} and "
            f"{FEETECH_GAIN_MAX} and must be an integer."
        )


@dataclass(frozen=True, kw_only=True)
class FeetechPIDGains:
    """Persistent position-loop gains stored by one Feetech actuator."""

    p: int
    i: int
    d: int

    def __post_init__(self) -> None:
        _validate_gain("P", self.p)
        _validate_gain("I", self.i)
        _validate_gain("D", self.d)


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
    max_position_velocity: float = DEFAULT_MAX_POSITION_VELOCITY
    position_tracking_error_gain: float = DEFAULT_POSITION_TRACKING_ERROR_GAIN
    position_pid: FeetechPIDGains | None = None

    def __post_init__(self) -> None:
        if not FEETECH_ACTUATOR_ID_MIN <= self.actuator_id <= FEETECH_ACTUATOR_ID_MAX:
            raise ValueError(
                "Feetech actuator ID must be between "
                f"{FEETECH_ACTUATOR_ID_MIN} and {FEETECH_ACTUATOR_ID_MAX}."
            )
        validate_feetech_acceleration(self.max_acceleration)
        if not math.isfinite(self.max_position_velocity) or self.max_position_velocity <= 0.0:
            raise ValueError("Feetech maximum position velocity must be positive and finite.")
        if (
            not math.isfinite(self.position_tracking_error_gain)
            or self.position_tracking_error_gain < 0.0
        ):
            raise ValueError(
                "Feetech position tracking-error gain must be finite and non-negative."
            )
