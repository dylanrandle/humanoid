"""Forwarding helpers consume runtime defaults without contacting devices or hosts."""

import os
import subprocess
from pathlib import Path

import pytest

from humanoid.constants import EnvironmentVariable, Host, Port

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _environment(tmp_path: Path, command: str) -> dict[str, str]:
    binary = tmp_path / command
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$FORWARD_TEST_CALLS"\n')
    binary.chmod(0o755)
    env = dict(os.environ)
    for variable in (
        EnvironmentVariable.QUEST_PORT,
        EnvironmentVariable.DASHBOARD_PORT,
        EnvironmentVariable.LOCAL_DASHBOARD_PORT,
        EnvironmentVariable.ROBOT_HOST,
    ):
        env.pop(variable, None)
    return {
        **env,
        "PATH": f"{tmp_path}:{env['PATH']}",
        "FORWARD_TEST_CALLS": str(tmp_path / "calls"),
    }


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("action", ["start", "stop"])
def test_quest_forwarding_uses_shared_defaults_and_overrides(tmp_path, custom, action):
    env = _environment(tmp_path, "adb")
    quest_port, dashboard_port = Port.QUEST_SERVER, Port.API
    if custom:
        quest_port, dashboard_port = 9101, 9102
        env[EnvironmentVariable.QUEST_PORT] = str(quest_port)
        env[EnvironmentVariable.DASHBOARD_PORT] = str(dashboard_port)
    result = subprocess.run(
        [PROJECT_ROOT / "scripts/forward-quest", action],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    mappings = [
        f"-d reverse tcp:{port} tcp:{port}"
        if action == "start"
        else f"-d reverse --remove tcp:{port}"
        for port in (quest_port, dashboard_port)
    ]
    assert (tmp_path / "calls").read_text().splitlines() == ["-d get-state", *mappings]


@pytest.mark.parametrize("custom", [False, True])
def test_dashboard_forwarding_uses_shared_defaults_and_overrides(tmp_path, custom):
    env = _environment(tmp_path, "ssh")
    remote_port = local_port = Port.API
    host: str = Host.ROBOT
    if custom:
        remote_port, local_port = 9102, 9103
        env[EnvironmentVariable.DASHBOARD_PORT] = str(remote_port)
        env[EnvironmentVariable.LOCAL_DASHBOARD_PORT] = str(local_port)
        host = "custom-robot"
        env[EnvironmentVariable.ROBOT_HOST] = host
    result = subprocess.run(
        [PROJECT_ROOT / "scripts/forward-ports"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        f"-L {Host.LOOPBACK}:{local_port}:{Host.LOOPBACK}:{remote_port} {host}"
        in (tmp_path / "calls").read_text()
    )
