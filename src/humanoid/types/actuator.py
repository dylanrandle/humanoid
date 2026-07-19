"""Configuration types shared by actuator hardware implementations."""

from dataclasses import dataclass
from enum import StrEnum


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
