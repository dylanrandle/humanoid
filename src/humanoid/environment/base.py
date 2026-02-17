from abc import ABC, abstractmethod
from typing import Any

from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.transition import Transition


class Environment(ABC):
    """Base environment class following gymnasium Env specification.

    This environment provides a standard interface for reinforcement learning
    tasks, with reset() and step() methods. Unlike gymnasium, it returns a
    Transition object instead of a tuple.
    """

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Observation:
        """Reset the environment to an initial state.

        Args:
            seed: Optional random seed for reproducibility
            options: Optional dictionary of environment-specific options

        Returns:
            Initial observation after reset
        """

    @abstractmethod
    def step(self, action: Action) -> Transition:
        """Execute one step in the environment.

        Args:
            action: Action to execute in the environment (joint or tool space)

        Returns:
            Transition containing observation, reward, is_done, is_truncated, and info
        """

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
