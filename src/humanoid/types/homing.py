from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class HomingPreset(StrEnum):
    """Named robot configurations available to operator controls."""

    HOME = "home"
    REST = "rest"


@dataclass
class HomingTarget:
    """Target joint configuration consumed by the homing node.

    Published by the requester *alongside* an ``OrchestratorEvent(REQUEST_HOMING)``
    so the homing node has a target to act on when the orchestrator flips the
    mode to HOMING.
    """

    timestamp: float
    target_position: np.ndarray
