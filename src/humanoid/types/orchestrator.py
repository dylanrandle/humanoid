from dataclasses import dataclass
from enum import StrEnum


class Mode(StrEnum):
    """Active control mode selected by the orchestrator.

    Each mode determines which per-source command topics are forwarded to the
    final ROBOT_* topics consumed by the robot driver / OSC.
    """

    IDLE = "idle"
    HOMING = "homing"
    OCULUS = "oculus"
    KEYBOARD = "keyboard"


@dataclass
class OrchestratorMode:
    timestamp: float
    mode: Mode


class EventKind(StrEnum):
    """Events that drive orchestrator FSM transitions."""

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
    :mod:`humanoid.orchestrator_client`.
    """

    timestamp: float
    kind: EventKind
