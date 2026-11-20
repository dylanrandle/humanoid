"""Resident workers fail together and are never restarted behind the policy."""

import signal
import subprocess
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from humanoid.config.serving import SERVING_CONFIG
from humanoid.serving.server import ModelServer
from humanoid.types.serving import ModelKind, ServingState, ServingStatus


@pytest.fixture
def server(monkeypatch):
    instance = ModelServer(SERVING_CONFIG)
    monkeypatch.setattr("humanoid.serving.server.os.killpg", MagicMock())
    yield instance
    instance.close()


def test_shutdown_kills_both_groups_even_when_launcher_exited_or_wait_times_out(
    server, monkeypatch
):
    first, second = MagicMock(pid=101), MagicMock(pid=102)
    first.poll.return_value = 0
    first.wait.side_effect = subprocess.TimeoutExpired("uv", 2)
    server._processes = {ModelKind.SEGMENTATION: first, ModelKind.GRASP: second}
    kill = MagicMock()
    monkeypatch.setattr("humanoid.serving.server.os.killpg", kill)
    server.close()
    assert [call.args for call in kill.call_args_list] == [
        (101, signal.SIGKILL),
        (102, signal.SIGKILL),
    ]
    second.wait.assert_called_once()
    assert server.status.state is ServingState.STOPPED


def test_memory_reserve_failure_stops_workers_and_does_not_retry(server, monkeypatch):
    server._status = ServingStatus(state=ServingState.READY, message="Models ready")
    monkeypatch.setattr(
        "humanoid.serving.server.psutil.virtual_memory", lambda: MagicMock(available=0)
    )
    stop = MagicMock()
    monkeypatch.setattr(server, "_stop_workers", stop)
    server.monitor()
    server.monitor()
    assert server.status.state is ServingState.FAILED
    assert "host memory" in server.status.message
    stop.assert_called_once()
    assert server._stop.is_set()


def test_dead_worker_stops_other_worker(server, monkeypatch):
    server.config = replace(SERVING_CONFIG, minimum_host_memory_bytes=0)
    process = MagicMock(returncode=7)
    process.poll.return_value = 7
    server._processes = {ModelKind.GRASP: process}
    stop = MagicMock()
    monkeypatch.setattr(server, "_stop_workers", stop)
    server.monitor()
    assert server.status.state is ServingState.FAILED
    assert "code 7" in server.status.message
    stop.assert_called_once()
