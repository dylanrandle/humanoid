"""Collection request deduplication and batch lifecycle."""

import os
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from humanoid.config.collection import DEFAULT_COLLECTION_SETTINGS
from humanoid.config.robot.so101 import SO101_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.constants import Topic
from humanoid.nodes import collection as collection_module
from humanoid.nodes.collection import CollectionNode
from humanoid.types.collection import (
    CollectionAction,
    CollectionReport,
    CollectionRequest,
    CollectionRunState,
    EpisodeOutcome,
)
from humanoid.types.logging import LoggingState
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotName


def make_node(monkeypatch):
    subscriber, client = MagicMock(), MagicMock()
    subscriber.receive.return_value = None
    client.robot_name = RobotName.TRISKEL
    client.robot_config = TRISKEL_CONFIG
    monkeypatch.setattr(collection_module, "Subscriber", MagicMock(return_value=subscriber))
    monkeypatch.setattr(collection_module, "SimulationClient", MagicMock(return_value=client))
    return CollectionNode(), subscriber, client


def test_prepare_uses_selected_task_and_seed_without_recording(monkeypatch):
    node, subscriber, client = make_node(monkeypatch)
    client.robot_name = SO101_CONFIG.name
    client.robot_config = SO101_CONFIG
    request = CollectionRequest(
        0, "scene", CollectionAction.PREPARE, replace(DEFAULT_COLLECTION_SETTINGS, seed=17)
    )
    subscriber.receive.side_effect = [request, request]
    node.step()
    node.step()
    assert node.status.state is CollectionRunState.IDLE
    assert node.status.phase == "scene_ready"
    client.reset.assert_called_once()
    assert client.reset.call_args.args[0].name == "empty"
    assert client.reset.call_args.args[0].robot_config is SO101_CONFIG
    assert request.settings is not None
    assert client.reset.call_args.args[1] == request.settings.seed
    client.close.assert_not_called()


def test_empty_scene_collection_request_fails_without_reset_or_recording(monkeypatch):
    node, _, client = make_node(monkeypatch)
    collector = MagicMock()
    monkeypatch.setattr(collection_module, "EpisodeCollector", collector)
    node._execute(CollectionRequest(0, "run", CollectionAction.START, DEFAULT_COLLECTION_SETTINGS))
    assert node.status.state is CollectionRunState.FAILED
    assert "no automated collection task" in node.status.message
    client.reset.assert_not_called()
    collector.assert_not_called()


def test_stop_before_start_prevents_a_delayed_start(monkeypatch):
    node, subscriber, _ = make_node(monkeypatch)
    subscriber.receive.side_effect = [
        CollectionRequest(0, "run", CollectionAction.STOP),
        CollectionRequest(0, "run", CollectionAction.START, DEFAULT_COLLECTION_SETTINGS),
    ]
    run = MagicMock()
    monkeypatch.setattr(node, "_execute", run)
    node.step()
    node.step()
    assert node.status.state is CollectionRunState.ABORTED
    run.assert_not_called()


def test_delayed_requests_cannot_replace_a_newer_session(monkeypatch):
    node, subscriber, _ = make_node(monkeypatch)
    subscriber.receive.side_effect = [
        CollectionRequest(2, "new", CollectionAction.STOP),
        CollectionRequest(1, "old", CollectionAction.START, DEFAULT_COLLECTION_SETTINGS),
        CollectionRequest(1.5, "old", CollectionAction.STOP),
    ]
    run = MagicMock()
    monkeypatch.setattr(node, "_execute", run)
    for _ in range(3):
        node.step()
    assert node.status.run_id == "new"
    run.assert_not_called()


def test_session_counts_only_valid_successes_and_advances_seeds(monkeypatch, tmp_path):
    node, _, client = make_node(monkeypatch)
    collector = MagicMock()
    collector.collect.side_effect = [
        CollectionReport(tmp_path, EpisodeOutcome.SUCCEEDED, False, "invalid", 2, 4),
        CollectionReport(tmp_path, EpisodeOutcome.FAILED, True, "failed", 2, 4),
        CollectionReport(tmp_path, EpisodeOutcome.SUCCEEDED, True, "ok", 2, 4),
    ]
    factory = MagicMock(return_value=collector)
    monkeypatch.setattr(collection_module, "EpisodeCollector", factory)
    node._execute(
        CollectionRequest(
            0,
            "run",
            CollectionAction.START,
            replace(DEFAULT_COLLECTION_SETTINGS, task="cleanup", seed=7, output=str(tmp_path)),
        )
    )
    assert node.status.state is CollectionRunState.SUCCEEDED
    assert node.status.successes == 1
    assert [call.args[0] for call in collector.collect.call_args_list] == [7, 8, 9]
    assert factory.call_args.args[2] == str(tmp_path)
    client.close.assert_not_called()


@pytest.mark.skipif(
    os.environ.get("HUMANOID_RUN_RENDER_TESTS") != "1", reason="Needs MuJoCo rendering."
)
def test_collection_stop_flushes_recording_and_keeps_processes_alive(
    running_stack, tmp_path, monkeypatch
):
    node = CollectionNode(lcm_url=running_stack.url)
    request = CollectionRequest(
        0,
        "cancel-run",
        CollectionAction.START,
        replace(DEFAULT_COLLECTION_SETTINGS, task="cleanup", output=str(tmp_path)),
    )
    progress = node._progress
    sent = False

    def cancel_after_sample():
        nonlocal sent
        progress()
        if node.status.samples and not sent:
            sent = True
            node.runtime.publisher.publish(
                CollectionRequest(0, request.run_id, CollectionAction.STOP),
                Topic.COLLECTION_REQUEST,
            )

    monkeypatch.setattr(node, "_progress", cancel_after_sample)
    try:
        node._execute(request)
        assert node.status.state is CollectionRunState.ABORTED, node.status
        assert node.runtime.logging_status is not None
        assert node.runtime.logging_status.state is LoggingState.STOPPED
        assert node.runtime.mode is Mode.IDLE
        assert list(tmp_path.glob("*/episode.json"))
        assert all(process.is_alive() for process in running_stack.processes)
    finally:
        node.on_close()
