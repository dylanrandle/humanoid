import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from humanoid.integrations.rax import runner


@pytest.fixture
def local_runner(tmp_path, monkeypatch):
    login = tmp_path / "auth.json"
    login.write_text("dummy-credential")
    monkeypatch.setattr(runner, "auth_path", lambda: login)
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/usr/local/bin/docker")
    monkeypatch.setattr(runner, "assets", lambda: tmp_path)
    return tmp_path


def test_preparation_keeps_login_out_of_artifacts_and_holds_out_formal_seeds(local_runner):
    directory = local_runner / "run"
    runner.prepare_run(directory, "toy_car_cleanup")
    config = json.loads((directory / "input.json").read_text())
    assert not set(config["exploration_seeds"]).intersection(
        range(config["formal_seed"], config["formal_seed"] + config["formal_episodes"])
    )
    assert config["sim_gpu"] == "external"
    assert (directory / "operator/.codex/auth.json").read_text() == ""
    assert runner.auth_path().read_text() == "dummy-credential"
    command = runner.docker_command(directory, "test-run", prepare_only=True)
    assert "--prepare-only" in command
    assert "--non-interactive" not in command
    assert any("auth.json,readonly" in arg for arg in command)
    assert "--privileged" not in command
    assert "seccomp=unconfined" not in command
    assert "dummy-credential" not in " ".join(command)


def test_interrupted_runner_removes_its_container(local_runner, monkeypatch):
    monkeypatch.setattr(runner, "find_data_root", lambda _: local_runner)
    monkeypatch.setattr(runner.subprocess, "call", Mock(side_effect=KeyboardInterrupt))
    cleanup = Mock()
    monkeypatch.setattr(runner.subprocess, "run", cleanup)
    monkeypatch.setattr(runner.signal, "signal", Mock())
    assert runner.run() != 0
    args = cleanup.call_args.args[0]
    assert args[1:3] == ["rm", "--force"]
    assert args[3].startswith("humanoid-rax-")


def test_missing_login_fails_before_starting_container(local_runner):
    runner.auth_path().unlink()
    with pytest.raises(RuntimeError, match="Log in"):
        runner.docker_command(Path("/unused"), "test", prepare_only=False)


def test_resume_keeps_existing_session_and_uses_a_new_container_name(local_runner, monkeypatch):
    directory = local_runner / "humanoid-rax-existing"
    runner.prepare_run(directory, "toy_car_cleanup")
    trial = directory / "trial"
    trial.mkdir()
    (trial / "agent-lifecycle.json").write_text('{"status":"stopped"}')
    memory = trial / "learned-program.py"
    memory.write_text("# Existing learned program and evidence must survive.\n")
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    launch = Mock(return_value=0)
    monkeypatch.setattr(runner.subprocess, "call", launch)
    monkeypatch.setattr(runner.subprocess, "run", Mock())
    monkeypatch.setattr(runner.signal, "signal", Mock())

    assert runner.run(resume=directory) == 0

    command = launch.call_args.args[0]
    assert "--resume" in command
    assert command[command.index("--name") + 1] != directory.name
    assert f"type=bind,source={directory},target=/home/rax/runtime" in command
    assert before == {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_resume_rejects_mismatched_task_and_prepare_only(local_runner, monkeypatch):
    directory = local_runner / "run"
    runner.prepare_run(directory, "different_task")
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "call", launch)
    with pytest.raises(ValueError, match="match the selected"):
        runner.run(resume=directory)
    with pytest.raises(ValueError, match="prepared-only"):
        runner.run(resume=directory, prepare_only=True)
    launch.assert_not_called()


def test_adaptation_uses_new_layouts_and_readonly_source_without_changing_history(local_runner):
    source = local_runner / "source"
    trial = source / "trial"
    trial.mkdir(parents=True)
    config = runner.configuration("toy_car_cleanup")
    config["exploration_seeds"] = [97, 98, 99]
    state = {"configuration": config, "active": None, "results": [{"task_complete": False}]}
    (trial / "state.json").write_text(json.dumps(state))
    before = (trial / "state.json").read_bytes()
    destination = local_runner / "next"

    runner.prepare_run(destination, "toy_car_cleanup", adapt_from=source)

    prepared = json.loads((destination / "input.json").read_text())
    assert prepared["exploration_seeds"] == [103, 104, 105, 106, 107, 108]
    assert prepared["adaptation_context"] == "compact"
    assert prepared["adaptation_source"] == "/home/rax/runtime/operator/source-trial"
    assert (trial / "state.json").read_bytes() == before
    command = runner.docker_command(destination, "transfer", prepare_only=True)
    assert (
        f"type=bind,source={trial},target=/home/rax/runtime/operator/source-trial,readonly"
        in command
    )
    assert (destination / "operator/.codex/auth.json").read_text() == ""


def test_adaptation_rejects_active_or_unrelated_trials_before_preparation(local_runner):
    source = local_runner / "source"
    trial = source / "trial"
    trial.mkdir(parents=True)
    config = runner.configuration("toy_car_cleanup")
    state = {"configuration": config, "active": {"episode": 1}}
    (trial / "state.json").write_text(json.dumps(state))
    destination = local_runner / "next"
    with pytest.raises(ValueError, match="active episode"):
        runner.prepare_run(destination, "toy_car_cleanup", adapt_from=source)
    assert not destination.exists()
    with pytest.raises(ValueError, match="match the selected"):
        runner.prepare_run(destination, "unrelated_task", adapt_from=source)
    assert not destination.exists()


def test_resume_and_adaptation_are_mutually_exclusive(local_runner):
    with pytest.raises(ValueError, match="either resume or adaptation"):
        runner.run(resume=local_runner, adapt_from=local_runner)


@pytest.fixture
def saved_controller(local_runner):
    source = local_runner / "source"
    trial = source / "trial"
    frozen = trial / "bundles/frozen"
    frozen.mkdir(parents=True)
    (frozen / "controller.py").write_text("def main(ctx): pass\n")
    state = {
        "configuration": runner.configuration("toy_car_cleanup"),
        "active": None,
        "bundles": {"saved": {"directory": "frozen", "sha256": "expected-digest"}},
    }
    (trial / "state.json").write_text(json.dumps(state))
    return source, state


def test_replay_preserves_source_and_needs_no_model_login(local_runner, saved_controller):
    source, _ = saved_controller
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    runner.auth_path().unlink()
    destination = local_runner / "replay"
    runner.prepare_replay(destination, "toy_car_cleanup", source, seed=23)
    command = runner.docker_command(destination, "replay", prepare_only=False)
    assert command[-1] == "/tmp/replay.py"
    assert "harness.codex_cli.auto_research" not in command
    assert not any("auth.json" in arg for arg in command)
    assert any("source-trial,readonly" in arg for arg in command)
    assert json.loads((destination / "input.json").read_text())["exploration_seeds"] == [23]
    assert json.loads((destination / "replay-source.json").read_text())["bundle"] == "saved"
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ["active", "task", "bundle", "escape", "seed"])
def test_replay_rejects_invalid_source_before_creating_run(local_runner, saved_controller, problem):
    source, state = saved_controller
    if problem == "active":
        state["active"] = {"episode": 1}
    elif problem == "task":
        state["configuration"]["task"] = "different"
    elif problem == "bundle":
        state["bundles"] = {}
    elif problem == "escape":
        state["bundles"]["saved"]["directory"] = ".."
    (source / "trial/state.json").write_text(json.dumps(state))
    destination = local_runner / "replay"
    with pytest.raises(ValueError):
        runner.prepare_replay(
            destination, "toy_car_cleanup", source, seed=-1 if problem == "seed" else 0
        )
    assert not destination.exists()


def test_contact_preview_uses_read_only_endpoint_and_single_formal_run(local_runner):
    directory = local_runner / "preview"
    runner.prepare_run(directory, runner.CONTACT_PREVIEW_TASK)
    command = runner.docker_command(directory, "preview", prepare_only=False)
    assert f"HUMANOID_RAX_TASK={runner.CONTACT_PREVIEW_TASK}" in command
    assert "HUMANOID_OBSERVATION_ENDPOINT" in command
    assert not any(
        arg.startswith(("HUMANOID_RAX_TOKEN", "HUMANOID_RAX_ENDPOINT")) for arg in command
    )
    config = json.loads((directory / "input.json").read_text())
    assert config["formal_episodes"] == 1
