"""Configuration types shared by actuator hardware implementations."""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

ActuatorEffortUnit = Literal["N·m", "N"]


class ActuatorEffortSource(StrEnum):
    CURRENT_ESTIMATE = "current_estimate"
    SIMULATION = "simulation"


@dataclass(frozen=True, kw_only=True)
class ActuatorEffortLimits:
    """Output effort ratings in N·m for rotary actuators or N for linear actuators."""

    stall: float
    rated: float | None = None
    unit: ActuatorEffortUnit = "N·m"

    def __post_init__(self) -> None:
        if not math.isfinite(self.stall) or self.stall <= 0.0:
            raise ValueError("Actuator stall effort must be positive and finite.")
        if self.rated is not None and (
            not math.isfinite(self.rated) or not 0.0 < self.rated <= self.stall
        ):
            raise ValueError("Actuator rated effort must be positive and no greater than stall.")


@dataclass(frozen=True, kw_only=True)
class ActuatorCurrentCalibration:
    """Convert signed current feedback units into estimated output effort."""

    amperes_per_unit: float
    effort_per_ampere: float

    def __post_init__(self) -> None:
        for value in (self.amperes_per_unit, self.effort_per_ampere):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("Actuator current calibration must be positive and finite.")


@dataclass(frozen=True, kw_only=True)
class ActuatorFeedback:
    """One controller's feedback, keyed by actuator ID; effort may be unavailable."""

    positions: dict[int, float]
    velocities: dict[int, float]
    temperatures: dict[int, float]
    # Signed SI effort in the same joint coordinates as position and velocity.
    efforts: dict[int, float] = field(default_factory=dict)


class ActuatorControlMode(StrEnum):
    """Command mode used for a configured actuator."""

    POSITION = "position"
    VELOCITY = "velocity"


@dataclass(frozen=True, kw_only=True)
class ActuatorControllerConfig:
    """Shared configuration for one actuator controller or bus."""


@dataclass(frozen=True, kw_only=True)
class ActuatorConfig:
    """Configuration owned by one physical actuator."""

    actuator_id: int
    controller: str
    inverted: bool = False
    effort_limits: ActuatorEffortLimits | None = None


@dataclass
class ActuatorHardwareConfig:
    """Physical actuator controllers and their joint bindings."""

    controllers: dict[str, ActuatorControllerConfig]
    joints: dict[str, ActuatorConfig]

    def __post_init__(self) -> None:
        seen_addresses: set[tuple[str, int]] = set()
        for joint_name, actuator in self.joints.items():
            if not joint_name:
                raise ValueError("Actuator joint names must not be empty.")
            controller = self.controllers.get(actuator.controller)
            if controller is None:
                raise ValueError(
                    f"Actuator for {joint_name} references unknown controller "
                    f"{actuator.controller!r}."
                )
            address = (actuator.controller, actuator.actuator_id)
            if address in seen_addresses:
                raise ValueError(
                    f"Duplicate actuator ID {actuator.actuator_id} on controller "
                    f"{actuator.controller!r}."
                )
            seen_addresses.add(address)


@dataclass(frozen=True)
class ActuatorHealth:
    """Latest health information for one physical actuator."""

    joint_name: str
    controller: str
    actuator_id: int
    healthy: bool
    temperature_celsius: float | None = None
    issue: str | None = None


@dataclass(frozen=True)
class ActuatorHealthReport:
    """Actuator telemetry emitted by the real robot driver."""

    timestamp: float
    actuators: tuple[ActuatorHealth, ...]
    error: str | None = None


@dataclass(frozen=True)
class ActuatorHealthStatus:
    """Freshness-qualified actuator health retained for the dashboard."""

    connected: bool
    healthy: bool
    age_seconds: float | None
    actuators: tuple[ActuatorHealth, ...]
    error: str | None = None
