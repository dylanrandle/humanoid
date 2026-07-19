from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from humanoid.types.homing import HomingPreset
from humanoid.types.logging import LoggingStatus
from humanoid.types.node import NodeRateStatus
from humanoid.types.process import ProcessName, ProcessStatus, Runtime
from humanoid.types.replay import RecordingSummary, ReplayStatus
from humanoid.types.robot import RobotName


class OrchestratorError(Exception):
    """An expected orchestrator-service failure."""


class Mode(StrEnum):
    """Active control mode selected by the orchestrator.

    Each mode determines which per-source command topics are forwarded to the
    final ROBOT_* topics consumed by the robot driver / OSC.
    """

    IDLE = "idle"
    HOMING = "homing"
    OCULUS = "oculus"
    KEYBOARD = "keyboard"


class OrchestratorParameter(StrEnum):
    PRESET = "preset"


_MODE_PARAMETER_TYPES: dict[Mode, dict[OrchestratorParameter, type[object]]] = {
    Mode.HOMING: {OrchestratorParameter.PRESET: HomingPreset},
}


@dataclass(frozen=True)
class OrchestratorRequest:
    """Request an orchestrator mode with validated, mode-specific parameters."""

    mode: Mode
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            parameters = {
                OrchestratorParameter(name): value for name, value in self.parameters.items()
            }
        except ValueError as exc:
            raise ValueError("Unknown orchestrator parameter.") from exc

        parameter_types = _MODE_PARAMETER_TYPES.get(self.mode, {})
        missing = parameter_types.keys() - parameters.keys()
        if missing:
            names = ", ".join(sorted(parameter.value for parameter in missing))
            raise ValueError(f"Mode {self.mode.value} requires parameters: {names}.")

        unexpected = parameters.keys() - parameter_types.keys()
        if unexpected:
            names = ", ".join(sorted(parameter.value for parameter in unexpected))
            raise ValueError(f"Mode {self.mode.value} does not accept parameters: {names}.")

        normalized: dict[OrchestratorParameter, object] = {}
        for name, expected_type in parameter_types.items():
            value = parameters[name]
            if not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {name.value} parameter for mode {self.mode.value}."
                    ) from exc
            normalized[name] = value
        object.__setattr__(self, "parameters", MappingProxyType(normalized))


@dataclass(frozen=True)
class ModeStatus:
    """Freshness-qualified mode broadcast observed by OrchestratorMonitor."""

    mode: Mode | None
    connected: bool
    age_seconds: float | None


@dataclass(frozen=True)
class OrchestratorState(ModeStatus):
    """Observed mode plus parameters for the active or pending request."""

    parameters: dict[str, object]


@dataclass(frozen=True)
class OrchestratorStatus:
    """Aggregate operator-console status produced by OrchestratorService."""

    runtime: Runtime
    robot: RobotName
    robots: list[RobotName]
    processes: dict[ProcessName, ProcessStatus]
    node_rates: list[NodeRateStatus]
    logging: LoggingStatus
    recordings: list[RecordingSummary]
    replay: ReplayStatus
    orchestrator: OrchestratorState


@dataclass(frozen=True)
class SafetyContext:
    """Configuration observed and hardware risk acknowledged by an operator."""

    expected_runtime: Runtime
    expected_robot: RobotName
    real_hardware_acknowledged: bool = False


@dataclass
class OrchestratorMode:
    timestamp: float
    mode: Mode


class EventKind(StrEnum):
    """Events that drive orchestrator mode transitions."""

    REQUEST_HOMING = "request_homing"
    REQUEST_OCULUS = "request_oculus"
    REQUEST_KEYBOARD = "request_keyboard"
    REQUEST_IDLE = "request_idle"
    COMPLETE = "complete"
    START_LOGGING = "start_logging"
    STOP_LOGGING = "stop_logging"


@dataclass
class OrchestratorEvent:
    """A pure signal published to ORCHESTRATOR_EVENT.

    Events that have parameters (e.g., ``REQUEST_HOMING`` needs a target) ship
    those parameters on a dedicated topic in the same call — see
    :mod:`humanoid.orchestrator.client`.
    """

    timestamp: float
    kind: EventKind
