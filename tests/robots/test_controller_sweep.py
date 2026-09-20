import json
from unittest.mock import MagicMock

import pytest

from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.hardware.actuators.feetech.config import FeetechActuatorConfig, FeetechPIDGains
from humanoid.robots.utils.controller_tracking import sweep
from humanoid.robots.utils.controller_tracking.models import ControllerTrackingSettings
from humanoid.robots.utils.controller_tracking.sweep import (
    load_sweep,
    resolve_robot_config,
    write_sweep_report,
)
from humanoid.types.controller_sweep import CompletedRun, DiagnosticInvocation


def _gains(p: int, i: int = 0, d: int = 32) -> FeetechPIDGains:
    return FeetechPIDGains(p=p, i=i, d=d)


def test_gain_sweep_supports_hardware_baseline_default_and_overrides(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {"label": "current", "gains": {}},
                    {
                        "label": "p24",
                        "low_acceleration_cost": 0.03,
                        "gains": {
                            "default": {"p": 24, "i": 0, "d": 32},
                            "2": {"p": 24, "i": 4, "d": 16},
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline = {1: _gains(16), 2: _gains(32, 4, 16)}

    runs = load_sweep(path, [1, 2])

    assert runs[0].resolve(baseline) == baseline
    assert runs[1].resolve(baseline) == {1: _gains(24), 2: _gains(24, 4, 16)}
    assert runs[0].low_acceleration_cost is None
    assert runs[1].low_acceleration_cost == pytest.approx(0.03)


@pytest.mark.parametrize(
    "payload",
    [
        {"runs": [{"label": "only", "gains": {}}]},
        {
            "runs": [
                {"label": "duplicate", "gains": {}},
                {"label": "duplicate", "gains": {}},
            ]
        },
        {
            "runs": [
                {"label": "baseline", "gains": {}},
                {"label": "bad", "gains": {"99": {"p": 1, "i": 2, "d": 3}}},
            ]
        },
        {
            "runs": [
                {"label": "baseline", "gains": {}},
                {"label": "bad", "gains": {}, "low_acceleration_cost": -0.1},
            ]
        },
    ],
)
def test_gain_sweep_rejects_invalid_run_sets(tmp_path, payload):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_sweep(path, [1, 2])


def test_feetech_pid_gains_reject_boolean_values():
    with pytest.raises(ValueError, match="P gain"):
        FeetechPIDGains(p=True, i=0, d=32)


def test_resolve_robot_config_applies_run_overrides_without_mutating_default():
    original_hardware = TRISKEL_CONFIG.hardware
    assert original_hardware is not None and original_hardware.actuators is not None
    original_arm_2 = original_hardware.actuators.joints["arm_2"]
    assert isinstance(original_arm_2, FeetechActuatorConfig)
    original_osc = TRISKEL_CONFIG.operational_space_config
    assert original_osc is not None

    resolved = resolve_robot_config(
        TRISKEL_CONFIG,
        {2: _gains(24, 0, 16)},
        low_acceleration_cost=0.07,
    )

    resolved_hardware = resolved.hardware
    assert resolved_hardware is not None and resolved_hardware.actuators is not None
    resolved_arm_2 = resolved_hardware.actuators.joints["arm_2"]
    assert isinstance(resolved_arm_2, FeetechActuatorConfig)
    assert resolved_arm_2.position_pid == _gains(24, 0, 16)
    assert resolved.operational_space_config is not None
    assert resolved.operational_space_config.low_acceleration_cost == pytest.approx(0.07)
    assert original_arm_2.position_pid != resolved_arm_2.position_pid
    assert TRISKEL_CONFIG.operational_space_config is original_osc


def test_run_injects_resolved_config_into_stack_and_diagnostic(monkeypatch, tmp_path):
    manager = MagicMock()
    manager_type = MagicMock(return_value=manager)
    diagnostic = MagicMock(return_value=tmp_path / "controller_tracking")
    monkeypatch.setattr(sweep, "NodeManager", manager_type)
    monkeypatch.setattr(sweep, "run_controller_tracking_diagnostic", diagnostic)
    settings = ControllerTrackingSettings()

    sweep._run_with_config(
        TRISKEL_CONFIG,
        diagnostic=DiagnosticInvocation(
            settings=settings,
            label="test",
            output_directory=tmp_path,
        ),
        startup_delay_s=0.0,
    )

    manager_type.assert_called_once_with(
        runtime=sweep.Runtime.REAL,
        robot_config=TRISKEL_CONFIG,
    )
    manager.start.assert_called_once_with(sweep.ProcessName.STACK)
    manager.wait_until_robot_ready.assert_called_once_with()
    diagnostic.assert_called_once_with(
        settings,
        TRISKEL_CONFIG,
        output_directory=tmp_path,
        label="test",
    )
    manager.close.assert_called_once_with()


def test_diagnostic_failure_reason_reads_partial_run_metadata(tmp_path):
    run = tmp_path / "controller_tracking_triskel_20260913_000000"
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "completed": False,
                "failure_reason": "Timed out waiting for measured HOME convergence",
            }
        ),
        encoding="utf-8",
    )

    assert sweep._diagnostic_failure_reason(tmp_path) == (
        "Timed out waiting for measured HOME convergence"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"completed": True, "failure_reason": None},
        {"completed": False, "failure_reason": ""},
        [],
    ],
)
def test_diagnostic_failure_reason_ignores_missing_reason(tmp_path, payload):
    run = tmp_path / "controller_tracking_triskel_20260913_000000"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")

    assert sweep._diagnostic_failure_reason(tmp_path) is None
    if isinstance(payload, dict) and payload.get("completed") is True:
        sweep._require_completed_diagnostic(tmp_path)
    else:
        with pytest.raises(RuntimeError, match=r"did not complete|readable metadata"):
            sweep._require_completed_diagnostic(tmp_path)


def test_controller_sweep_report_compares_tracking_and_smoothness(tmp_path):
    baseline = _completed_run(
        tmp_path,
        "baseline",
        tracking_rms=0.1,
        shake_rms=0.02,
        low_acceleration_cost=0.01,
    )
    tuned = _completed_run(
        tmp_path,
        "tuned",
        tracking_rms=0.05,
        shake_rms=0.01,
        low_acceleration_cost=0.03,
    )
    report = tmp_path / "comparison.md"

    write_sweep_report(report, [baseline, tuned])

    contents = report.read_text(encoding="utf-8")
    assert "Actuator 1" in contents
    assert "16/0/32" in contents
    assert "Low-acceleration cost" in contents
    assert "| baseline | 16/0/32 | 0.01 |" in contents
    assert "| tuned | 16/0/32 | 0.03 |" in contents
    assert "Joint tracking RMS (rad)" in contents
    assert "High-frequency joint motion RMS (rad)" in contents
    assert "Commanded joint acceleration RMS (rad/s²)" in contents
    assert "Commanded joint acceleration p95 (rad/s²)" in contents
    assert "0.05 (-50.0%)" in contents
    assert "OSC translation tracking RMS" in contents
    assert "End-to-end translation tracking RMS" in contents


def _completed_run(
    root,
    label: str,
    *,
    tracking_rms: float,
    shake_rms: float,
    low_acceleration_cost: float | None = None,
) -> CompletedRun:
    run_directory = root / label
    run_directory.mkdir()
    diagnostic_directory = run_directory / "controller_tracking_triskel_20260913_000000"
    diagnostic_directory.mkdir()
    metrics_path = diagnostic_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "tracking": {
                    "overall": {
                        "joints": {"arm_1": {"rms": tracking_rms}},
                        "osc_tool_translation_m": {"rms": tracking_rms},
                        "osc_tool_orientation_rad": {"rms": tracking_rms},
                        "end_to_end_tool_translation_m": {"rms": tracking_rms},
                        "end_to_end_tool_orientation_rad": {"rms": tracking_rms},
                    }
                },
                "smoothness": {
                    "overall": {
                        "joints": {
                            "arm_1": {
                                "high_frequency_rms_rad": shake_rms,
                                "jerk_rms_rad_s3": shake_rms,
                                "acceleration_p95_rad_s2": shake_rms,
                                "command_acceleration_rms_rad_s2": shake_rms,
                                "command_acceleration_p95_rad_s2": shake_rms,
                            }
                        },
                        "tool_high_frequency_rms_m": shake_rms,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return CompletedRun(
        label=label,
        gains={1: _gains(16)},
        metrics_path=metrics_path,
        run_directory=run_directory,
        low_acceleration_cost=low_acceleration_cost,
    )
