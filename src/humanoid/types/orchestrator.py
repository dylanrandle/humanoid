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
