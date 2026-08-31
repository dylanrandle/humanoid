"""Behavior checks for Raspberry Pi helper commands in the root launcher."""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRISKEL = REPO_ROOT / "triskel"


def _fake_command(directory: Path, name: str) -> None:
    command = directory / name
    command.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    command.chmod(0o755)


def _run_helper(tmp_path: Path, *arguments: str, **environment: str) -> subprocess.CompletedProcess:
    for command in ("ssh", "rsync"):
        _fake_command(tmp_path, command)
    env = os.environ.copy()
    env.pop("TRISKEL_SSH_TARGET", None)
    env.pop("TRISKEL_REMOTE_ROOT", None)
    env.update(environment)
    env["PATH"] = f"{tmp_path}:{os.environ['PATH']}"
    return subprocess.run(
        [TRISKEL, *arguments],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_dashboard_forwards_both_operator_ports(tmp_path: Path):
    result = _run_helper(tmp_path, "dashboard", "robot@triskel.local")

    assert result.returncode == 0, result.stderr
    assert "Dashboard:     http://127.0.0.1:8765" in result.stdout
    assert "Visualization: http://127.0.0.1:8080" in result.stdout
    assert "-L\n8765:127.0.0.1:8765" in result.stdout
    assert "-L\n8080:127.0.0.1:8080" in result.stdout
    assert result.stdout.rstrip().endswith("robot@triskel.local")


def test_recordings_uses_environment_defaults_and_portable_rsync_flags(tmp_path: Path):
    result = _run_helper(
        tmp_path,
        "recordings",
        TRISKEL_SSH_TARGET="robot@triskel.local",
        TRISKEL_REMOTE_ROOT="/srv/triskel",
    )

    assert result.returncode == 0, result.stderr
    assert "robot@triskel.local:/srv/triskel/recordings/" in result.stdout
    assert str(REPO_ROOT / "recordings") + "/" in result.stdout
    for option in ("--archive", "--human-readable", "--partial", "--progress"):
        assert option in result.stdout


def test_recordings_defaults_to_home_checkout(tmp_path: Path):
    result = _run_helper(tmp_path, "recordings", "robot@triskel.local")

    assert result.returncode == 0, result.stderr
    assert "robot@triskel.local:~/humanoid/recordings/" in result.stdout


def test_remote_helpers_require_an_ssh_target(tmp_path: Path):
    result = _run_helper(tmp_path, "dashboard", TRISKEL_SSH_TARGET="")

    assert result.returncode == 1
    assert "Pass an SSH target or set TRISKEL_SSH_TARGET" in result.stderr


def test_recordings_rejects_remote_shell_metacharacters(tmp_path: Path):
    result = _run_helper(
        tmp_path,
        "recordings",
        "robot@triskel.local",
        TRISKEL_REMOTE_ROOT="~/humanoid;echo-unsafe",
    )

    assert result.returncode == 1
    assert "TRISKEL_REMOTE_ROOT must be a simple absolute or ~/ path" in result.stderr
