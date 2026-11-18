"""Operator lifecycle and run status for autonomy integrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from humanoid.types.camera import CameraFrames
from humanoid.types.rax import RaxCommandResult, RaxOperation
from humanoid.types.remote_control import RemoteControlPolicyConfig
from humanoid.types.robot import RobotState
from humanoid.types.tasks import TaskEvaluation

if TYPE_CHECKING:
    from humanoid.robots.base import Robot


class RobotExecution(Protocol):
    """A trusted backend; task objectives and agent code do not live here."""

    robot: Robot
    step_id: int
    config: RemoteControlPolicyConfig

    def wait_until_ready(self, timeout: float = 30.0) -> None: ...
    def state(self) -> RobotState: ...
    def observe(self) -> CameraFrames: ...
    def execute(
        self, operation: RaxOperation | str, arguments: Mapping[str, object]
    ) -> RaxCommandResult: ...
    def hold(self) -> None: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


class EpisodeLifecycle(Protocol):
    """Operator-owned lifecycle; reset and evaluation are never robot tools."""

    def start_episode(self, *, formal: bool, seed: int) -> str: ...
    def create_execution(self) -> RobotExecution: ...
    def evaluate_episode(self) -> TaskEvaluation: ...
    def close_episode(self) -> None: ...


@dataclass(frozen=True, kw_only=True)
class AutonomyStatus:
    """Operator-facing run state, separate from agent observations."""

    task: str | None
    tasks: tuple[str, ...]
    running: bool
    ready: bool
    message: str
