from dataclasses import dataclass, field
from typing import TypedDict

from humanoid.types.observation import Observation


class TransitionInfo(TypedDict):
    """Information dictionary for transitions.

    All fields are required to ensure consistent information across environments.
    """

    command_timestamp: float
    observation_timestamp: float
    latency: float


@dataclass
class Transition:
    observation: Observation
    reward: float
    is_done: bool
    is_truncated: bool
    info: TransitionInfo = field(default_factory=dict)
