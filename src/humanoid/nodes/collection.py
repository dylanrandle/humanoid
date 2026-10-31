"""Operator-requested collection as an independent client of the normal stack."""

import time
from dataclasses import replace

from humanoid.config import ROBOT_CONFIG
from humanoid.config.collection import DEFAULT_COLLECTION_CONFIG
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.base import Node
from humanoid.simulation.client import SimulationClient
from humanoid.simulation.collection.collector import EpisodeCollector
from humanoid.simulation.tasks.base import SimulationTask
from humanoid.simulation.tasks.registry import TASKS
from humanoid.types.collection import (
    CollectionAction,
    CollectionCancelled,
    CollectionRequest,
    CollectionRunState,
    CollectionRunStatus,
    CollectionSettings,
    EpisodeOutcome,
)
from humanoid.types.robot import RobotConfig

STATUS_INTERVAL_SECONDS = 0.2


class CollectionNode(Node):
    rate_hz = 10.0

    def __init__(self, robot_config: RobotConfig = ROBOT_CONFIG, lcm_url: str = DEFAULT_LCM_URL):
        self.runtime = SimulationClient(robot_config, lcm_url)
        self.subscriber = Subscriber([Topic.COLLECTION_REQUEST], url=lcm_url, queue_size=None)
        self.status = CollectionRunStatus(time.time())
        self._last_published = 0.0
        self._last_request = float("-inf")
        self._cancelled = False
        self._collector: EpisodeCollector | None = None

    def setup(self) -> None:
        self.publish(force=True)

    def step(self) -> None:
        request = self.subscriber.receive(Topic.COLLECTION_REQUEST)
        if (
            request is not None
            and request.run_id != self.status.run_id
            and request.timestamp > self._last_request
        ):
            self._last_request = request.timestamp
            if request.action is CollectionAction.STOP:
                # Remember an early Stop so a delayed/retried Start cannot run it.
                self.status = CollectionRunStatus(
                    time.time(),
                    CollectionRunState.ABORTED,
                    request.run_id,
                    message="Collection stopped before the first attempt.",
                )
            else:
                self._execute(request)
        self.publish()

    def _execute(self, request: CollectionRequest) -> None:
        settings = request.settings
        preparing = request.action is CollectionAction.PREPARE
        self.status = CollectionRunStatus(
            time.time(),
            CollectionRunState.PREPARING if preparing else CollectionRunState.STARTING,
            request.run_id,
        )
        self._cancelled = False
        self.runtime.before_step = self._check_cancelled
        self.publish(force=True)
        try:
            if settings is None or settings.task not in TASKS:
                raise ValueError("Select a registered collection task.")
            task = TASKS[settings.task](robot_config=self.runtime.robot_config)
            if not preparing and not task.collectable:
                raise ValueError(f"{task.label} has no automated collection task.")
            self.status = replace(self.status, task=settings.task, seed=settings.seed)
            if preparing:
                self.runtime.reset(task, settings.seed, DEFAULT_COLLECTION_CONFIG)
                self.status = replace(
                    self.status,
                    state=CollectionRunState.IDLE,
                    phase="scene_ready",
                    message=f"{task.label} scene loaded with seed {settings.seed}.",
                )
            else:
                self._run(task, settings)
        except CollectionCancelled:
            self.status = replace(
                self.status,
                state=CollectionRunState.ABORTED,
                message="Scene loading stopped." if preparing else "Collection stopped.",
            )
        except Exception as error:
            self.status = replace(
                self.status,
                state=CollectionRunState.FAILED,
                message=f"{type(error).__name__}: {error}",
            )
        finally:
            self.runtime.before_step = None
            self.publish(force=True)

    def publish(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_published >= STATUS_INTERVAL_SECONDS:
            self.status = replace(self.status, timestamp=time.time())
            self.runtime.publisher.publish(self.status, Topic.COLLECTION_STATUS)
            self._last_published = now

    def _check_cancelled(self) -> None:
        while (request := self.subscriber.receive(Topic.COLLECTION_REQUEST)) is not None:
            if request.run_id == self.status.run_id and request.action is CollectionAction.STOP:
                self._last_request = max(self._last_request, request.timestamp)
                self._cancelled = True
                self.status = replace(self.status, state=CollectionRunState.STOPPING)
                self.publish(force=True)
        # Keep the UI informed during reset and logger waits as well as expert ticks.
        self.publish()
        if self._cancelled:
            raise CollectionCancelled("Collection stopped by the operator.")

    def _progress(self) -> None:
        logging = self.runtime.logging_status
        self.status = replace(
            self.status,
            phase=self.runtime.phase,
            target=self.runtime.target,
            sim_seconds=self.runtime.time,
            actions=self.runtime.action_count,
            samples=self._collector.sample_count if self._collector else 0,
            recording=(logging.file_name or "") if logging else "",
        )
        self.publish()

    def _run(self, task: SimulationTask, settings: CollectionSettings) -> None:
        try:
            self.status = replace(
                self.status,
                state=CollectionRunState.RUNNING,
                episodes=settings.episodes,
                max_attempts=settings.max_attempts,
            )
            config = replace(DEFAULT_COLLECTION_CONFIG, max_sim_seconds=settings.max_sim_seconds)
            collector = self._collector = EpisodeCollector(
                task, self.runtime, settings.output, config
            )
            self.runtime.on_tick = self._progress
            for attempt in range(settings.max_attempts):
                self._check_cancelled()
                self.status = replace(
                    self.status,
                    attempts=attempt + 1,
                    seed=settings.seed + attempt,
                    phase="reset",
                    actions=0,
                    samples=0,
                    sim_seconds=0.0,
                    message="",
                    recording="",
                    target="",
                )
                self.publish(force=True)
                report = collector.collect(settings.seed + attempt)
                self.status = replace(
                    self.status,
                    successes=self.status.successes + int(report.eligible_for_training),
                    recording=str(report.directory),
                    samples=report.sample_count,
                    actions=report.action_count,
                    message=" ".join((report.reason, *report.errors)),
                )
                if report.outcome is EpisodeOutcome.ABORTED:
                    self.status = replace(self.status, state=CollectionRunState.ABORTED)
                    break
                if self.status.successes >= settings.episodes:
                    self.status = replace(self.status, state=CollectionRunState.SUCCEEDED)
                    break
            else:
                self.status = replace(
                    self.status,
                    state=CollectionRunState.FAILED,
                    message=f"Collected {self.status.successes}/{settings.episodes} successful "
                    f"episodes within {settings.max_attempts} attempts. {self.status.message}",
                )
        finally:
            self.runtime.on_tick = None
            self.runtime.observe = None
            self._collector = None

    def on_close(self) -> None:
        self.subscriber.close()
        self.runtime.close()
