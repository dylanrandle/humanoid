from abc import ABC, abstractmethod

from humanoid.types.action import Action
from humanoid.types.observation import Observation


class Policy(ABC):
    """Base policy class for generating actions from observations."""

    @abstractmethod
    def __call__(self, observation: Observation) -> Action:
        """Generate an action given an observation.

        Args:
            observation: Current observation from the environment

        Returns:
            Action to execute in the environment
        """

    def reset(self) -> None:  # noqa: B027
        """Reset policy state (optional, for stateful policies)."""
