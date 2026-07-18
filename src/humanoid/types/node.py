"""Node lifecycle types."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.process import BaseProcess

    from humanoid.nodes.base import Node
    from humanoid.types.process import ProcessName, Runtime
    from humanoid.types.robot import RobotName


class ProcessContext(Protocol):
    def Process(
        self,
        *,
        target: Callable[..., None],
        name: str,
    ) -> BaseProcess: ...


@dataclass(frozen=True)
class NodeGroup:
    name: ProcessName
    display_name: str
    nodes: tuple[type[Node], ...]
    deferred_nodes: tuple[type[Node], ...] = ()
    allowed_runtimes: frozenset[Runtime] | None = None


@dataclass
class ManagedNodeGroup:
    runtime: Runtime
    robot: RobotName
    started_monotonic: float
    processes: list[BaseProcess] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    startup_thread: threading.Thread | None = None
    ended_monotonic: float | None = None
    failure_exit_code: int | None = None
    last_output: str | None = None
    stop_requested: bool = False
    shutdown_started: bool = False
