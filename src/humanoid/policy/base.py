from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from humanoid.types.action import Action
from humanoid.types.observation import Observation
from humanoid.types.orchestrator import Mode


class Policy(ABC):
    """Base policy class for generating actions from observations.

    Subclasses set ``mode`` to the orchestrator mode under which they should
    run, and implement :meth:`step`. The first active call runs :meth:`on_enter`
    before stepping. The base class gates execution: when the
    observation reports a mode that differs from ``self.mode``, the policy is
    reset (so its internal references re-anchor on next activation) and an
    empty Action is returned — the environment publishes nothing for empty
    fields, so the orchestrator forwards nothing.

    Observations with ``mode=None`` (e.g. tests, or running without an
    orchestrator) are treated as active so policies still work standalone.
    """

    mode: ClassVar[Mode]
    _active = False
    _previous_mode_command_time: float | None = None

    def __call__(self, observation: Observation) -> Action:
        if observation.mode is not None and observation.mode != self.mode:
            self._active = False
            self.reset()
            return Action()
        if not self._active:
            self.on_enter(observation)
            self._active = True
        return self.step(observation)

    def on_enter(self, observation: Observation) -> None:
        """Mark the cached joint command as belonging to the previous mode."""
        command = observation.robot_joint_command
        self._previous_mode_command_time = command.timestamp if command is not None else None

    def _base_reference_positions(self, observation: Observation) -> NDArray[np.float64]:
        """The controller starts at measured odometry, then integrates its commands."""
        command = observation.robot_joint_command
        if command is not None and (
            self._previous_mode_command_time is None
            or command.timestamp > self._previous_mode_command_time
        ):
            return command.joint_positions
        return observation.robot_state.joint_positions

    @abstractmethod
    def step(self, observation: Observation) -> Action:
        """Compute an action from an observation (called only when active)."""

    def reset(self) -> None:  # noqa: B027
        """Reset policy state (optional, for stateful policies)."""
