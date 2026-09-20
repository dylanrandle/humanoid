"""Checks for experiment timing, references, and acceptance measurements."""

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.osc_tuning import BASELINE_OSC_CONFIG, tuning_scenarios
from humanoid.robots.utils.osc_tuning import (
    analyze_trial,
    reference_at,
    run_scenario,
    save_manifest,
)


@pytest.mark.parametrize(
    "name", ["home_translation_ramp", "offset_orientation_ramp", "unseen_reference_jitter"]
)
def test_ramp_feedforward_matches_pose_derivative(name):
    scenarios = tuning_scenarios("screen") + tuning_scenarios("validation")
    scenario = next(s for s in scenarios if s.name == name)
    anchor = pin.SE3.Random()
    step = 1e-5
    for elapsed in (0.0, 0.2, 0.7, 1.0, 2.0):
        left, _ = reference_at(scenario, anchor, elapsed - step)
        right, _ = reference_at(scenario, anchor, elapsed + step)
        _, velocity = reference_at(scenario, anchor, elapsed)
        np.testing.assert_allclose(
            velocity.linear,
            (right.translation - left.translation) / (2 * step),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            velocity.angular,
            pin.log3(right.rotation @ left.rotation.T) / (2 * step),
            atol=1e-8,
        )


@pytest.fixture(scope="module")
def ramp_trial():
    scenario = next(s for s in tuning_scenarios("screen") if s.name == "home_translation_ramp")
    config = replace(BASELINE_OSC_CONFIG, manipulability_cost=0.0)
    trace, metadata = run_scenario(config, scenario)
    return trace, metadata, config, scenario


def test_native_simulation_meets_gates_and_preserves_command_ownership(ramp_trial):
    fk_tolerance = 1e-10
    trace, metadata, config, scenario = ramp_trial
    metrics = analyze_trial(trace, metadata, config, scenario)
    assert metrics["passed"], metrics["failed_gates"]
    np.testing.assert_allclose(np.unique(np.round(trace["dt"][1:], 3)), [0.03, 0.035])
    uncontrolled = np.setdiff1d(np.arange(trace["command_q"].shape[1]), metadata["arm_q_indices"])
    np.testing.assert_allclose(
        trace["command_q"][:, uncontrolled],
        np.broadcast_to(
            np.array(metadata["initial_q"])[uncontrolled],
            (len(trace["command_q"]), len(uncontrolled)),
        ),
    )
    position_error, orientation_error = metrics["native_fk_max_m"], metrics["native_fk_max_rad"]
    assert isinstance(position_error, float) and position_error < fk_tolerance
    assert isinstance(orientation_error, float) and orientation_error < fk_tolerance


def test_raw_command_spike_and_solver_failure_cannot_win_on_smoothness(ramp_trial):
    source, metadata, config, scenario = ramp_trial
    trace = {key: values.copy() for key, values in source.items()}
    trace["command_v"][20, metadata["arm_v_indices"][0]] += 0.5
    trace["solve_success"][20] = 0
    trace["clearance"][30] = 0.001
    metrics = analyze_trial(trace, metadata, config, scenario)
    assert not metrics["passed"]
    failed_gates = metrics["failed_gates"]
    assert isinstance(failed_gates, list)
    assert {"acceleration_limits", "solver", "clearance"} <= set(failed_gates)


def test_validation_is_disjoint_from_screening():
    train = tuning_scenarios("training")
    screen = tuning_scenarios("screen")
    validation = tuning_scenarios("validation")
    confirmation = tuning_scenarios("confirmation")
    assert set(screen) <= set(train)
    assert not {s.arm_positions for s in train} & {s.arm_positions for s in validation}
    assert not {s.arm_positions for s in train + validation} & {
        s.arm_positions for s in confirmation
    }


def test_manifest_reconstructs_staged_unstaged_and_untracked_sources(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source = repo / "src"
    source.mkdir(parents=True)

    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)

    git("init", "--quiet")
    tracked = source / "tracked.py"
    tracked.write_text("value = 0\n")
    git("add", "src/tracked.py")
    git(
        "-c",
        "user.name=OSC manifest test",
        "-c",
        "user.email=osc-manifest@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--quiet",
        "-m",
        "Baseline",
    )
    tracked.write_text("value = 1\n")
    staged = source / "staged.py"
    staged.write_text("added = True\n")
    git("add", "src")
    tracked.write_text("value = 2\n")
    (source / "untracked.py").write_text("untracked = True\n")
    output = tmp_path / "experiment"
    output.mkdir()
    monkeypatch.setattr("humanoid.robots.utils.osc_tuning.find_repo_root", lambda _: repo)

    save_manifest(output, BASELINE_OSC_CONFIG, ())

    replay = tmp_path / "replay"
    git("clone", "--quiet", "--no-hardlinks", str(repo), str(replay))
    subprocess.run(
        ["git", "apply", str(output / "source.patch")],
        cwd=replay,
        check=True,
        capture_output=True,
    )
    shutil.copytree(output / "untracked_sources", replay, dirs_exist_ok=True)
    manifest = json.loads((output / "manifest.json").read_text())
    for original in source.iterdir():
        name = original.relative_to(repo).as_posix()
        restored = (replay / name).read_bytes()
        assert restored == original.read_bytes()
        assert hashlib.sha256(restored).hexdigest() == manifest["source_sha256"][name]
