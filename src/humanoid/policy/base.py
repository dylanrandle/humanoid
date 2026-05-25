from abc import ABC, abstractmethod
from typing import ClassVar

from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode


class Policy(ABC):
    """Base policy class for generating actions from observations.

    Subclasses set ``mode`` to the orchestrator mode under which they should
    run, and implement :meth:`step`. The base class gates execution: when the
    observation reports a mode that differs from ``self.mode``, the policy is
    reset (so its internal references re-anchor on next activation) and an
    empty Action is returned — the environment publishes nothing for empty
    fields, so the orchestrator forwards nothing.

    Observations with ``mode=None`` (e.g. tests, or running without an
    orchestrator) are treated as active so policies still work standalone.
    """

    mode: ClassVar[Mode]

    def __call__(self, observation: Observation) -> Action:
        if observation.mode is not None and observation.mode != self.mode:
            self.reset()
            return Action()
        return self.step(observation)

    @abstractmethod
    def step(self, observation: Observation) -> Action:
        """Compute an action from an observation (called only when active)."""

    def reset(self) -> None:  # noqa: B027
        """Reset policy state (optional, for stateful policies)."""
