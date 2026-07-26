"""Node lifecycle types."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.process import BaseProcess

    from humanoid.nodes.base import Node
    from humanoid.types.process import ProcessName, Runtime
    from humanoid.types.robot import RobotName
    from humanoid.types.simulation import MujocoScene


class ProcessContext(Protocol):
    def Process(
        self,
        *,
        target: Callable[..., None],
        name: str,
    ) -> BaseProcess: ...


@dataclass(frozen=True)
class NodeRateSample:
    """Loop-rate measurement published by a running node process."""

    timestamp: float
    node_name: str
    pid: int
    target_rate_hz: float
    measured_rate_hz: float


@dataclass(frozen=True)
class NodeRateStatus:
    """Freshness-qualified node-rate health shown in the operator console."""

    node_name: str
    pid: int
    target_rate_hz: float | None
    measured_rate_hz: float | None
    healthy: bool
    age_seconds: float | None


@dataclass(frozen=True)
class NodeGroup:
    name: ProcessName
    display_name: str
    nodes: tuple[type[Node], ...]
    deferred_nodes: tuple[type[Node], ...] = ()
    runtime_nodes: Mapping[Runtime, tuple[type[Node], ...]] = field(default_factory=dict)

    def nodes_for_runtime(self, runtime: Runtime) -> tuple[type[Node], ...]:
        """Return runtime-specific nodes followed by the group's common nodes."""
        return (*self.runtime_nodes.get(runtime, ()), *self.nodes)


@dataclass
class ManagedNodeGroup:
    runtime: Runtime
    robot: RobotName
    scene: MujocoScene
    started_monotonic: float
    processes: list[BaseProcess] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    startup_thread: threading.Thread | None = None
    ended_monotonic: float | None = None
    failure_exit_code: int | None = None
    last_output: str | None = None
    stop_requested: bool = False
    shutdown_started: bool = False
