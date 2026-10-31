"""Collect timestamped observations using the operator's asynchronous stack."""

import json
import subprocess
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import mujoco

from humanoid.config.collection import COLLECTION_SCHEMA_VERSION, DEFAULT_COLLECTION_CONFIG
from humanoid.constants import Topic
from humanoid.simulation.camera import SimulationCamera
from humanoid.simulation.client import SimulationClient
from humanoid.simulation.collection.log import contains_episode_event, validate_collection_log
from humanoid.simulation.tasks.base import SimulationTask
from humanoid.types.collection import (
    CollectionCancelled,
    CollectionConfig,
    CollectionEpisode,
    CollectionReport,
    EpisodeOutcome,
    SimulationSample,
)
from humanoid.types.logging import LoggingRequest, LoggingState
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotJointCommand, RobotState
from humanoid.types.task_annotation import TaskAnnotation
from humanoid.utils.paths import find_repo_root


def _provenance() -> dict[str, object]:
    root = find_repo_root(__file__)

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "mujoco_version": mujoco.__version__,
    }


class EpisodeCollector:
    """Own episode metadata and cameras, never the controller, driver, or logger."""

    def __init__(
        self,
        task: SimulationTask,
        runtime: SimulationClient,
        output: str | Path,
        config: CollectionConfig = DEFAULT_COLLECTION_CONFIG,
    ):
        self.task, self.runtime, self.output, self.config = task, runtime, str(output), config
        self.sample_count = 0

    def collect(self, seed: int) -> CollectionReport:  # noqa: PLR0915 - episode lifecycle
        if not self.task.collectable:
            raise ValueError(f"{self.task.label} has no automated collection task.")
        if seed < 0:
            raise ValueError("Episode seed must be nonnegative.")
        runtime = self.runtime
        runtime.reset(self.task, seed, self.config)
        engine, publisher = runtime.engine, runtime.publisher
        episode_id = str(uuid.uuid4())
        self.sample_count = 0
        cameras: list[SimulationCamera] = []
        metadata = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "seed": seed,
            "observation_modalities": ["rgb"],
            "camera_names": list(self.task.cameras()),
            "observation_validation": "task_checks_per_sample",
            "action_alignment": "asynchronous_last_applied_command",
            "action_count": "policy_targets",
            "scheduling": "asynchronous",
            "control_path": [
                "SYSTEM",
                "OrchestratorNode",
                "RobotControllerNode",
                "MujocoSimulationNode",
            ],
            "collection": asdict(self.config),
            "task_config": self.task.metadata(),
            "provenance": _provenance(),
        }

        def episode(outcome: EpisodeOutcome, reason: str = "") -> CollectionEpisode:
            return CollectionEpisode(
                time.perf_counter(),
                episode_id,
                self.task.name,
                self.task.version,
                Runtime.SIM,
                "host_monotonic",
                outcome,
                self.sample_count,
                runtime.action_count,
                json.dumps(
                    {**metadata, "reason": reason, "final_task_state": self.task.state(runtime)},
                    allow_nan=False,
                ),
            )

        last_sample = float("-inf")
        started_at = float("inf")

        def observe(state: RobotState, command: RobotJointCommand, _index: int) -> None:
            nonlocal last_sample
            logging = runtime.logging_status
            if logging is None or logging.state is not LoggingState.RUNNING:
                raise RuntimeError(
                    (logging.error if logging else "") or "LCM recording stopped unexpectedly."
                )
            if (
                state.timestamp < started_at
                or state.timestamp - last_sample < 1 / self.config.sample_hz
            ):
                return
            sample = SimulationSample(
                state.timestamp,
                episode_id,
                self.sample_count,
                -1,
                state,
                command,
                tuple(camera.capture(state.timestamp) for camera in cameras),
                tuple(
                    replace(item, timestamp=state.timestamp) for item in self.task.objects(engine)
                ),
                TaskAnnotation(
                    state.timestamp,
                    episode_id,
                    self.task.name,
                    self.task.version,
                    _sample_labels(self.task, runtime),
                ),
            )
            publisher.publish(sample, Topic.SIMULATION_SAMPLE)
            self.sample_count += 1
            last_sample = state.timestamp

        interrupted = False
        try:
            for name, camera in self.task.cameras().items():
                cameras.append(SimulationCamera(engine, name, camera))
            if not cameras:
                raise ValueError("A collection task must expose at least one observation camera.")
            path = _start_logger(
                runtime, self.output, episode(EpisodeOutcome.READY), self.config.startup_timeout
            )
            started = episode(EpisodeOutcome.STARTED)
            started_at = started.timestamp
            publisher.publish(started, Topic.COLLECTION_EPISODE)
            runtime.observe = observe
            try:
                outcome, reason = _run_expert(self.task, runtime)
            except KeyboardInterrupt:
                interrupted = True
                outcome, reason = EpisodeOutcome.ABORTED, "Collection interrupted."
            finalized = _record_terminal(
                runtime, path, episode(outcome, reason), self.config.finalization_timeout
            )
        finally:
            runtime.observe = None
            with ExitStack() as cleanup:
                for camera in cameras:
                    cleanup.callback(camera.close)
                cleanup.callback(_stop_logger, runtime, self.config.finalization_timeout)
                cleanup.callback(runtime.stop)
        errors = validate_collection_log(path, episode_id)
        if not finalized:
            errors += ("Final episode event was not completely flushed before shutdown.",)
        if (
            runtime.logging_status is not None
            and runtime.logging_status.state is LoggingState.FAILED
        ):
            errors += (runtime.logging_status.error or "LCM logger failed.",)
        report = CollectionReport(
            path.parent,
            outcome,
            not errors,
            reason,
            self.sample_count,
            runtime.action_count,
            errors,
        )
        summary = {
            **metadata,
            "episode_id": episode_id,
            "runtime": Runtime.SIM.value,
            "clock": "host_monotonic",
            "task": self.task.name,
            "task_version": self.task.version,
            "outcome": outcome.value,
            "reason": reason,
            "valid": report.valid,
            "eligible_for_training": report.eligible_for_training,
            "samples": self.sample_count,
            "actions": runtime.action_count,
            "sim_seconds": runtime.time,
            "validation_errors": errors,
            "final_task_state": self.task.state(runtime),
        }
        path.with_name("episode.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        if interrupted:
            raise KeyboardInterrupt
        return report


def _stop_logger(runtime: SimulationClient, timeout: float) -> None:
    request = LoggingRequest(time.perf_counter(), False)
    requested = time.time()
    deadline, next_retry = time.monotonic() + timeout, 0.0
    while time.monotonic() < deadline:
        if time.monotonic() >= next_retry:
            runtime.publisher.publish(request, Topic.LOGGING_REQUEST)
            next_retry = time.monotonic() + 0.25
        runtime.poll()
        status = runtime.logging_status
        if (
            status is not None
            and status.timestamp > requested
            and status.state in {LoggingState.STOPPED, LoggingState.FAILED}
        ):
            return
        time.sleep(0.02)
    raise TimeoutError("The stack logger did not stop the collection recording.")


def _sample_labels(task: SimulationTask, runtime: SimulationClient) -> str:
    errors = task.observation_errors(runtime)
    return json.dumps(
        {**task.state(runtime), "observation_valid": not errors, "observation_errors": errors},
        allow_nan=False,
    )


def _record_terminal(
    runtime: SimulationClient, path: Path, event: CollectionEpisode, timeout: float
) -> bool:
    tail = path.stat().st_size
    runtime.publisher.publish(event, Topic.COLLECTION_EPISODE)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if contains_episode_event(path, event.episode_id, event.outcome, offset=tail):
            return True
        runtime.poll()
        status = runtime.logging_status
        if status is None or status.state is not LoggingState.RUNNING:
            break
        # Continuous robot traffic triggers lcm-logger's periodic flush.
        time.sleep(0.02)
    return False


def _start_logger(
    runtime: SimulationClient, output: str, probe: CollectionEpisode, timeout: float
) -> Path:
    request = LoggingRequest(time.perf_counter(), True, output)
    requested = time.time()
    deadline, next_retry = time.monotonic() + timeout, 0.0
    while time.monotonic() < deadline:
        runtime._check_cancelled()
        if time.monotonic() >= next_retry:
            runtime.publisher.publish(request, Topic.LOGGING_REQUEST)
            next_retry = time.monotonic() + 0.25
        runtime.poll()
        status = runtime.logging_status
        if status is not None and status.timestamp > requested:
            if status.state is LoggingState.FAILED:
                raise RuntimeError(status.error)
            if status.state is LoggingState.RUNNING and status.file_name:
                path = Path(status.file_name)
                runtime.publisher.publish(probe, Topic.COLLECTION_EPISODE)
                if contains_episode_event(path, probe.episode_id, EpisodeOutcome.READY):
                    return path
        time.sleep(0.02)
    raise TimeoutError("LCM logger startup probe was not recorded.")


def _run_expert(task: SimulationTask, runtime: SimulationClient) -> tuple[EpisodeOutcome, str]:
    try:
        runtime.activate()
        result = task.run(runtime)
        verified = result.success and task.success(runtime.engine)
        outcome = EpisodeOutcome.SUCCEEDED if verified else EpisodeOutcome.FAILED
        reason = (
            result.reason
            if not result.success or verified
            else "Task returned success but final-state validation failed."
        )
        return outcome, reason
    except CollectionCancelled:
        return EpisodeOutcome.ABORTED, "Collection interrupted."
    except Exception as error:
        return EpisodeOutcome.FAILED, f"{type(error).__name__}: {error}"
