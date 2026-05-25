from dataclasses import dataclass

import numpy as np


@dataclass
class HomingTarget:
    """Target joint configuration consumed by the homing node.

    Published by the requester *alongside* an ``OrchestratorEvent(REQUEST_HOMING)``
    so the homing node has a target to act on when the orchestrator flips the
    mode to HOMING.
    """

    timestamp: float
    target_position: np.ndarray
