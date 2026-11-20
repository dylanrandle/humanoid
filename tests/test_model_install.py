"""Deployment updates worker code without resolving or replacing CUDA dependencies."""

import subprocess
from unittest.mock import MagicMock

import pytest

from humanoid.utils.model_install import install_worker_package


def test_shared_worker_environment_is_updated_once_without_dependencies(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.touch()
    wheel = tmp_path / "humanoid-0.2.0-py3-none-any.whl"
    wheel.touch()
    pythons = (python, python)
    run = MagicMock()
    monkeypatch.setattr("humanoid.utils.model_install.subprocess.run", run)
    install_worker_package(wheel, pythons, "/configured/uv")
    run.assert_called_once()
    command = run.call_args.args[0]
    assert command[:3] == ["/configured/uv", "pip", "install"]
    assert command[command.index("--python") + 1] == str(python)
    assert "--no-deps" in command
    assert command[-1] == str(wheel)
    assert run.call_args.kwargs["check"] is True


def test_missing_environments_are_skipped_but_install_failures_abort(tmp_path, monkeypatch):
    python = tmp_path / "python"
    wheel = tmp_path / "humanoid-0.2.0-py3-none-any.whl"
    wheel.touch()
    pythons = (python, python)
    run = MagicMock(side_effect=subprocess.CalledProcessError(1, "uv"))
    monkeypatch.setattr("humanoid.utils.model_install.subprocess.run", run)
    install_worker_package(wheel, pythons, "uv")
    run.assert_not_called()
    python.touch()
    with pytest.raises(subprocess.CalledProcessError):
        install_worker_package(wheel, pythons, "uv")
    with pytest.raises(ValueError, match="wheel"):
        install_worker_package(tmp_path / "missing.whl", pythons, "uv")
