"""Collection lifecycle, asynchronous timing, and opt-in real LCM/RGB recording."""

import json
import os
import time
from contextlib import closing
from dataclasses import replace
from io import BytesIO
from unittest.mock import MagicMock

import lcm
import numpy as np
import pytest
from PIL import Image

from humanoid.config.collection import COLLECTION_SCHEMA_VERSION, DEFAULT_COLLECTION_CONFIG
from humanoid.constants import Topic
from humanoid.recording import RecordingCatalog
from humanoid.simulation.collection.collector import EpisodeCollector, _run_expert
from humanoid.simulation.tasks.cleanup.task import CleanupTask
from humanoid.simulation.tasks.empty import EmptySceneTask
from humanoid.simulation.tasks.registry import TASKS
from humanoid.types.collection import CollectionCancelled, EpisodeOutcome, TaskResult
from humanoid.types.lcm import simulation_sample_t
from humanoid.types.lcm.collection import sample_from_lcm
from humanoid.types.logging import LoggingState
from humanoid.types.orchestrator import Mode

MIN_SUCCESSFUL_SAMPLE_COUNT = 100


def test_task_registry():
    assert set(TASKS) == {"empty", "cleanup"}


def test_empty_scene_cannot_start_an_episode(tmp_path):
    runtime = MagicMock()
    collector = EpisodeCollector(EmptySceneTask(), runtime, tmp_path)
    with pytest.raises(ValueError, match="no automated collection task"):
        collector.collect(0)
    assert runtime.mock_calls == []
    assert list(tmp_path.iterdir()) == []


def test_false_success_is_rejected(monkeypatch):
    task = CleanupTask()
    engine = task.create_engine(0)
    runtime = MagicMock(engine=engine)
    monkeypatch.setattr(task, "run", lambda _: TaskResult(True, "Unverified claim."))
    outcome, reason = _run_expert(task, runtime)
    assert outcome is EpisodeOutcome.FAILED
    assert "validation failed" in reason


@pytest.mark.parametrize(
    "error,outcome",
    [
        (CollectionCancelled(), EpisodeOutcome.ABORTED),
        (RuntimeError("fault"), EpisodeOutcome.FAILED),
    ],
)
def test_exception_outcomes_are_explicit(monkeypatch, error, outcome):
    task = CleanupTask()
    runtime = MagicMock(engine=task.create_engine(0))

    def fail(_):
        raise error

    monkeypatch.setattr(task, "run", fail)
    assert _run_expert(task, runtime)[0] is outcome


def test_activation_failure_is_reported_without_running_the_expert():
    task, runtime = MagicMock(), MagicMock()
    runtime.activate.side_effect = TimeoutError("The orchestrator did not enter system mode.")
    outcome, reason = _run_expert(task, runtime)
    assert outcome is EpisodeOutcome.FAILED
    assert "did not enter system mode" in reason
    task.run.assert_not_called()


@pytest.mark.skipif(
    os.environ.get("HUMANOID_RUN_RENDER_TESTS") != "1",
    reason="Needs MuJoCo graphics and local LCM multicast.",
)
def test_complete_collection_uses_existing_lcm_bundles_with_decodable_rgb(
    tmp_path, episode_collector
):
    task = CleanupTask()
    report = episode_collector(task).collect(0)
    assert report.eligible_for_training, report
    bundle = RecordingCatalog(tmp_path).get(report.directory.name)
    manifest = json.loads((bundle.directory / "episode.json").read_text())
    assert manifest["schema_version"] == COLLECTION_SCHEMA_VERSION
    assert manifest["observation_modalities"] == ["rgb"]
    assert manifest["camera_names"] == list(task.cameras())
    assert manifest["eligible_for_training"] is True
    assert manifest["final_task_state"]["contained"] == list(task.bounds)
    assert report.sample_count > MIN_SUCCESSFUL_SAMPLE_COUNT
    assert report.action_count > report.sample_count
    inspected = 0
    with closing(lcm.EventLog(str(bundle.log_path), "r")) as log:
        while (event := log.read_next_event()) is not None:
            if event.channel != Topic.SIMULATION_SAMPLE.value:
                continue
            sample = sample_from_lcm(simulation_sample_t.decode(event.data))
            assert sample.state.timestamp == sample.timestamp
            assert sample.command.timestamp <= sample.timestamp
            assert sample.action_index == -1
            assert [observation.frame.name for observation in sample.cameras] == list(
                task.cameras()
            )
            for observation in sample.cameras:
                camera = observation.frame
                assert camera.timestamp == sample.timestamp
                rgb = np.asarray(Image.open(BytesIO(camera.rgb_jpeg)))
                config = task.cameras()[camera.name]
                assert rgb.shape == (config.height, config.width, 3)
            assert len(sample.objects) == len(task.bounds)
            labels = json.loads(sample.task_annotation.labels_json)
            assert labels["observation_valid"]
            assert labels["observation_errors"] == []
            assert all(view["camera"] in task.cameras() for view in labels["views"].values())
            for item in sample.objects:
                assert np.isfinite(item.pose.homogeneous).all()
                assert item.bbox_half_size is not None
                assert (item.bbox_half_size > 0).all()
            inspected += 1
    assert inspected == report.sample_count


@pytest.mark.skipif(
    os.environ.get("HUMANOID_RUN_RENDER_TESTS") != "1",
    reason="Needs MuJoCo graphics and local LCM multicast.",
)
def test_failed_episode_is_logged_but_excluded_from_training(episode_collector):
    config = replace(DEFAULT_COLLECTION_CONFIG, max_sim_seconds=1.0)
    report = episode_collector(CleanupTask(), config).collect(1)
    assert report.outcome is EpisodeOutcome.FAILED
    assert report.valid, report.errors
    assert not report.eligible_for_training
    assert "time budget" in report.reason


@pytest.mark.skipif(
    os.environ.get("HUMANOID_RUN_RENDER_TESTS") != "1",
    reason="Needs MuJoCo graphics, MeshCat, and local LCM multicast.",
)
def test_collection_reuses_running_nodes_and_leaves_stack_operational(
    episode_collector, running_stack
):
    config = replace(DEFAULT_COLLECTION_CONFIG, max_sim_seconds=1.0)
    collector = episode_collector(CleanupTask(), config)
    runtime = collector.runtime
    pids = [process.pid for process in running_stack.processes]
    for seed in (0, 1):
        report = collector.collect(seed)
        assert report.valid, report.errors
        assert [process.pid for process in running_stack.processes] == pids
        assert all(process.is_alive() for process in running_stack.processes)
        assert runtime.mode is Mode.IDLE
        before = runtime.time
        time.sleep(0.1)
        runtime.poll()
        assert runtime.time > before


@pytest.mark.skipif(
    os.environ.get("HUMANOID_RUN_RENDER_TESTS") != "1", reason="Needs graphics and multicast."
)
def test_cancelling_logger_startup_does_not_leave_a_delayed_start(episode_collector, monkeypatch):
    collector = episode_collector(CleanupTask())
    requested = False
    publish = collector.runtime.publisher.publish

    def watch(data, topic):
        nonlocal requested
        publish(data, topic)
        if topic is Topic.LOGGING_REQUEST and data.start:
            requested = True

    def cancel_after_start_request():
        if requested:
            raise CollectionCancelled

    monkeypatch.setattr(collector.runtime.publisher, "publish", watch)
    collector.runtime.before_step = cancel_after_start_request
    with pytest.raises(CollectionCancelled):
        collector.collect(0)
    time.sleep(0.3)
    collector.runtime.poll()
    assert collector.runtime.logging_status.state in {LoggingState.STOPPED, LoggingState.FAILED}
