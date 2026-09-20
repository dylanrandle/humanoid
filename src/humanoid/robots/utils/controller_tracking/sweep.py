"""Run the real-robot tracking diagnostic over controller configuration sweeps."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from humanoid.config.robot.triskel import (
    MAIN_CONTROLLER,
    POSITION_PID_GAINS_BY_ACTUATOR_ID,
    TRISKEL_CONFIG,
)
from humanoid.constants import Topic
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
    FeetechPIDGains,
)
from humanoid.hardware.actuators.feetech.configurator import FeetechActuatorConfigurator
from humanoid.logger import get_logger, setup_logging
from humanoid.middleware.subscriber import Subscriber
from humanoid.nodes.manager import NodeManager
from humanoid.orchestrator.constants import EXTERNAL_STACK_ERROR
from humanoid.robots.utils.controller_tracking.cli import (
    controller_tracking_settings_from_arguments,
    run_controller_tracking_diagnostic,
)
from humanoid.robots.utils.controller_tracking.metadata import json_ready
from humanoid.robots.utils.controller_tracking.models import FIGURE_EIGHT_SETTINGS
from humanoid.types.controller_sweep import (
    CompletedRun,
    DashboardState,
    DiagnosticInvocation,
    JointMetricSpec,
    SweepRunSpec,
)
from humanoid.types.process import ProcessName, Runtime
from humanoid.types.robot import RobotConfig
from humanoid.utils.paths import find_data_root

logger = get_logger(__name__)

DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8765"
DEFAULT_STACK_STARTUP_DELAY_SECONDS = 1.0
EXTERNAL_STACK_PROBE_SECONDS = 0.75
HTTP_TIMEOUT_SECONDS = 5.0
DASHBOARD_RESTORE_TIMEOUT_SECONDS = 5.0
DASHBOARD_RESTORE_POLL_SECONDS = 0.1
MINIMUM_COMPARISON_RUNS = 2
LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
MANAGED_TRACKING_OPTIONS = ("--compare-to", "--label", "--output-dir", "--robot")
PHASES = (
    ("overall", "Overall Cartesian figure-eight matrix"),
    *(
        (
            setting.name,
            f"{setting.plane.upper()} plane, {setting.size_multiplier:g}x size",
        )
        for setting in FIGURE_EIGHT_SETTINGS
    ),
)


def _default_output_directory() -> Path:
    return find_data_root(__file__) / "logs" / "tracking"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-robot tracking diagnostic over Feetech and OSC controller "
            "configurations, then compare the results. Any unrecognized arguments are "
            "forwarded to controller_tracking."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sweep_file",
        type=Path,
        help="JSON file describing the controller configurations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_directory(),
        help="Parent directory for the timestamped controller sweep",
    )
    parser.add_argument(
        "--dashboard-url",
        default=DEFAULT_DASHBOARD_URL,
        help="Operator-console URL; an active dashboard stack is stopped and restored",
    )
    parser.add_argument(
        "--stack-startup-delay",
        type=float,
        default=DEFAULT_STACK_STARTUP_DELAY_SECONDS,
        help="Extra delay after the first robot state before starting each diagnostic",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge the real-hardware motion and EEPROM writes without prompting",
    )
    return parser


def load_sweep(
    path: Path,
    actuator_ids: Sequence[int],
) -> list[SweepRunSpec]:
    """Load and validate every controller configuration before touching hardware."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read controller sweep file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Controller sweep file is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"runs"}:
        raise ValueError("Controller sweep JSON must contain exactly one top-level 'runs' array.")
    raw_runs = payload["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) < MINIMUM_COMPARISON_RUNS:
        raise ValueError("Controller sweep must contain at least two runs for comparison.")

    allowed_ids = set(actuator_ids)
    runs = [_parse_sweep_run(raw_run, allowed_ids) for raw_run in raw_runs]
    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("Controller sweep run labels must be unique.")
    return runs


def _parse_sweep_run(raw_run: object, actuator_ids: set[int]) -> SweepRunSpec:
    required_fields = {"label", "gains"}
    allowed_fields = {*required_fields, "low_acceleration_cost"}
    if (
        not isinstance(raw_run, dict)
        or not required_fields <= set(raw_run)
        or not set(raw_run) <= allowed_fields
    ):
        raise ValueError(
            "Each controller sweep run must contain 'label' and 'gains', with an optional "
            "'low_acceleration_cost'."
        )
    run_mapping = cast(dict[str, object], raw_run)
    label = run_mapping["label"]
    if not isinstance(label, str) or LABEL_PATTERN.fullmatch(label) is None:
        raise ValueError(
            "Controller sweep labels must be 1-64 characters using letters, digits, '.', "
            "'_', or '-'."
        )
    raw_gains = run_mapping["gains"]
    if not isinstance(raw_gains, dict):
        raise ValueError(f"Controller sweep run {label!r} must provide a gains object.")

    default = None
    overrides: dict[int, FeetechPIDGains] = {}
    for raw_actuator_id, raw_gain in raw_gains.items():
        if raw_actuator_id == "default":
            default = _parse_gains(raw_gain, f"run {label!r} default")
            continue
        if not isinstance(raw_actuator_id, str) or not raw_actuator_id.isdecimal():
            raise ValueError(
                f"Gain keys in run {label!r} must be 'default' or decimal actuator IDs."
            )
        actuator_id = int(raw_actuator_id)
        if str(actuator_id) != raw_actuator_id or actuator_id not in actuator_ids:
            raise ValueError(f"Unknown actuator ID {raw_actuator_id!r} in run {label!r}.")
        overrides[actuator_id] = _parse_gains(
            raw_gain,
            f"run {label!r}, actuator {actuator_id}",
        )
    raw_low_acceleration_cost = run_mapping.get("low_acceleration_cost")
    if raw_low_acceleration_cost is None:
        low_acceleration_cost = None
    elif (
        not isinstance(raw_low_acceleration_cost, int | float)
        or isinstance(raw_low_acceleration_cost, bool)
        or not math.isfinite(raw_low_acceleration_cost)
        or raw_low_acceleration_cost < 0.0
    ):
        raise ValueError(
            f"Low-acceleration cost for run {label!r} must be finite and non-negative."
        )
    else:
        low_acceleration_cost = float(raw_low_acceleration_cost)
    return SweepRunSpec(
        label=label,
        default=default,
        overrides=overrides,
        low_acceleration_cost=low_acceleration_cost,
    )


def _parse_gains(raw_gain: object, location: str) -> FeetechPIDGains:
    if not isinstance(raw_gain, dict) or set(raw_gain) != {"p", "i", "d"}:
        raise ValueError(f"PID gains for {location} must contain exactly p, i, and d.")
    gain_mapping = cast(dict[str, object], raw_gain)
    values = (gain_mapping["p"], gain_mapping["i"], gain_mapping["d"])
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError(f"PID gains for {location} must be integers.")
    try:
        return FeetechPIDGains(
            p=cast(int, values[0]),
            i=cast(int, values[1]),
            d=cast(int, values[2]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid PID gains for {location}: {exc}") from exc


def _validate_tracking_arguments(arguments: Sequence[str]) -> list[str]:
    forwarded = list(arguments)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    for argument in forwarded:
        if any(
            argument == option or argument.startswith(f"{option}=")
            for option in MANAGED_TRACKING_OPTIONS
        ):
            raise ValueError(
                f"{argument.split('=', maxsplit=1)[0]} is managed by the sweep utility."
            )
    return forwarded


def _confirm_hardware_sweep(runs: Sequence[SweepRunSpec], *, assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Run interactively or pass --yes to acknowledge hardware motion.")
    print(
        f"\nThis will run {len(runs)} full controller diagnostics on REAL hardware.\n"
        "The arm and gripper will move, and changed PID gains will be written to servo EEPROM.\n"
        "Any OSC low-acceleration costs apply only to their individual run.\n"
        "The gains present before the sweep will be restored afterward.\n"
    )
    response = input("Type RUN to continue: ").strip()
    if response != "RUN":
        raise RuntimeError("Controller sweep cancelled.")


def _controller_config() -> FeetechActuatorControllerConfig:
    hardware = TRISKEL_CONFIG.hardware
    if hardware is None or hardware.actuators is None:
        raise RuntimeError("Triskel has no actuator hardware configuration.")
    return cast(
        FeetechActuatorControllerConfig,
        hardware.actuators.controllers[MAIN_CONTROLLER],
    )


def _read_current_gains(
    actuator_ids: Sequence[int],
) -> dict[int, FeetechPIDGains]:
    controller_config = _controller_config()
    gains = {}
    for actuator_id in actuator_ids:
        values = FeetechActuatorConfigurator.read_gains(
            actuator_id,
            controller_config=controller_config,
        )
        gains[actuator_id] = FeetechPIDGains(**values)
    return gains


def resolve_robot_config(
    base: RobotConfig,
    gains: Mapping[int, FeetechPIDGains],
    *,
    low_acceleration_cost: float | None,
) -> RobotConfig:
    """Build one complete run configuration without mutating project defaults."""
    hardware = base.hardware
    actuator_hardware = hardware.actuators if hardware is not None else None
    if hardware is None or actuator_hardware is None:
        raise ValueError(f"Robot {base.name.value!r} has no actuator hardware configuration.")

    configured_ids: set[int] = set()
    joints = {}
    for joint_name, actuator in actuator_hardware.joints.items():
        gain = gains.get(actuator.actuator_id)
        if gain is None:
            joints[joint_name] = actuator
            continue
        if not isinstance(actuator, FeetechActuatorConfig):
            raise ValueError(f"Actuator {actuator.actuator_id} is not a Feetech position actuator.")
        joints[joint_name] = replace(actuator, position_pid=gain)
        configured_ids.add(actuator.actuator_id)

    missing_ids = set(gains) - configured_ids
    if missing_ids:
        missing = ", ".join(str(actuator_id) for actuator_id in sorted(missing_ids))
        raise ValueError(
            f"Robot {base.name.value!r} has no configured actuators with IDs: {missing}."
        )

    operational_space_config = base.operational_space_config
    if low_acceleration_cost is not None:
        if operational_space_config is None:
            raise ValueError(f"Robot {base.name.value!r} has no operational-space configuration.")
        operational_space_config = replace(
            operational_space_config,
            low_acceleration_cost=low_acceleration_cost,
        )

    return replace(
        base,
        hardware=replace(
            hardware,
            actuators=replace(actuator_hardware, joints=joints),
        ),
        operational_space_config=operational_space_config,
    )


def _run_with_config(
    robot_config: RobotConfig,
    *,
    diagnostic: DiagnosticInvocation | None,
    startup_delay_s: float,
) -> None:
    """Run one diagnostic with a complete config injected into every robot node."""
    manager = NodeManager(runtime=Runtime.REAL, robot_config=robot_config)
    try:
        manager.start(ProcessName.STACK)
        manager.wait_until_robot_ready()
        if startup_delay_s > 0.0:
            time.sleep(startup_delay_s)
        if diagnostic is None:
            return
        run_directory = run_controller_tracking_diagnostic(
            diagnostic.settings,
            robot_config,
            output_directory=diagnostic.output_directory,
            label=diagnostic.label,
        )
        if run_directory is None:
            raise RuntimeError("Controller tracking diagnostic was cancelled.")
    finally:
        manager.close()


def _find_metrics(run_directory: Path) -> Path:
    candidates = sorted(run_directory.rglob("metrics.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one metrics JSON in {run_directory}, found {len(candidates)}."
        )
    return candidates[0]


def _diagnostic_metadata(run_directory: Path) -> dict[str, object] | None:
    """Read the metadata emitted by one diagnostic invocation."""
    candidates = sorted(run_directory.rglob("metadata.json"))
    if len(candidates) != 1:
        return None
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast(dict[str, object], payload) if isinstance(payload, dict) else None


def _diagnostic_failure_reason(run_directory: Path) -> str | None:
    """Read the structured reason saved by a partially completed diagnostic."""
    payload = _diagnostic_metadata(run_directory)
    if payload is None:
        return None
    failure_reason = payload.get("failure_reason")
    return failure_reason if isinstance(failure_reason, str) and failure_reason else None


def _require_completed_diagnostic(run_directory: Path) -> None:
    """Reject missing or explicitly incomplete diagnostic metadata."""
    payload = _diagnostic_metadata(run_directory)
    if payload is None:
        raise RuntimeError(f"Expected one readable metadata JSON in {run_directory}.")
    if payload.get("completed") is not True:
        reason = _diagnostic_failure_reason(run_directory)
        detail = f": {reason}" if reason is not None else ""
        raise RuntimeError(f"Controller tracking diagnostic did not complete{detail}.")


def _dashboard_request(
    base_url: str,
    path: str,
    *,
    payload: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=({"Content-Type": "application/json"} if data is not None else {}),
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        try:
            error_payload = json.loads(exc.read())
            detail = error_payload.get("error", str(exc))
        except (json.JSONDecodeError, AttributeError):
            detail = str(exc)
        raise RuntimeError(f"Dashboard request {path} failed: {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Dashboard request {path} returned an invalid response.")
    return result


def _prepare_dashboard(base_url: str) -> DashboardState | None:
    try:
        snapshot = _dashboard_request(base_url, "/api/status")
    except URLError:
        logger.info("No operator console found at %s; continuing without it", base_url)
        return None

    processes = snapshot.get("processes")
    if not isinstance(processes, dict):
        raise RuntimeError("Dashboard status did not include process state.")
    stack = processes.get(ProcessName.STACK.value)
    if not isinstance(stack, dict):
        raise RuntimeError("Dashboard status did not include main-stack state.")
    other_running = [
        name
        for name, status in processes.items()
        if name != ProcessName.STACK.value
        and isinstance(status, dict)
        and status.get("running") is True
    ]
    if other_running:
        names = ", ".join(sorted(other_running))
        raise RuntimeError(
            f"Stop dashboard-managed processes before the controller sweep: {names}."
        )

    stack_was_running = stack.get("running") is True
    orchestrator = snapshot.get("orchestrator")
    mode = orchestrator.get("mode") if isinstance(orchestrator, dict) else None
    if stack_was_running and mode != "idle":
        raise RuntimeError(
            f"Put the dashboard stack in idle before the controller sweep (current mode: {mode})."
        )
    runtime = snapshot.get("runtime")
    robot = snapshot.get("robot")
    if not isinstance(runtime, str) or not isinstance(robot, str):
        raise RuntimeError("Dashboard status did not include its selected runtime and robot.")

    if stack_was_running:
        logger.info("Stopping the dashboard-managed stack for the controller sweep")
        _dashboard_request(base_url, "/api/processes/stack/stop", payload={})
    return DashboardState(
        base_url=base_url,
        stack_was_running=stack_was_running,
        runtime=runtime,
        robot=robot,
    )


def _restore_dashboard(state: DashboardState | None) -> None:
    if state is None or not state.stack_was_running:
        return
    logger.info("Restoring the dashboard-managed stack")
    deadline = time.monotonic() + DASHBOARD_RESTORE_TIMEOUT_SECONDS
    while True:
        try:
            _dashboard_request(state.base_url, "/api/status")
            break
        except RuntimeError as exc:
            if EXTERNAL_STACK_ERROR not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(DASHBOARD_RESTORE_POLL_SECONDS)
    _dashboard_request(
        state.base_url,
        "/api/processes/stack/start",
        payload={
            "expected_runtime": state.runtime,
            "expected_robot": state.robot,
            "real_hardware_acknowledged": True,
        },
    )


def _assert_no_external_stack() -> None:
    subscriber = Subscriber(topics=[Topic.ORCHESTRATOR_MODE])
    deadline = time.monotonic() + EXTERNAL_STACK_PROBE_SECONDS
    try:
        while time.monotonic() < deadline:
            if subscriber.receive(Topic.ORCHESTRATOR_MODE) is not None:
                raise RuntimeError(
                    "Another stack is still broadcasting. Stop it before running the "
                    "controller sweep."
                )
            time.sleep(0.05)
    finally:
        subscriber.close()


def _gains_json(gains: Mapping[int, FeetechPIDGains]) -> dict[str, dict[str, int]]:
    return {
        str(actuator_id): {"p": gain.p, "i": gain.i, "d": gain.d}
        for actuator_id, gain in sorted(gains.items())
    }


def _write_manifest(  # noqa: PLR0913 - manifest fields are explicit at each checkpoint
    path: Path,
    *,
    sweep_file: Path,
    original_gains: Mapping[int, FeetechPIDGains] | None,
    run_records: Sequence[Mapping[str, object]],
    status: str,
    error: str | None = None,
) -> None:
    payload = {
        "status": status,
        "error": error,
        "sweep_file": str(sweep_file.resolve()),
        "original_gains": _gains_json(original_gains) if original_gains is not None else None,
        "runs": list(run_records),
    }
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_sweep_report(path: Path, runs: Sequence[CompletedRun]) -> None:
    """Write a compact multi-run tracking and smoothness comparison."""
    metrics = [json.loads(run.metrics_path.read_text(encoding="utf-8")) for run in runs]
    lines = [
        "# Controller tracking tuning sweep",
        "",
        (
            "The first run is the baseline. Parenthesized percentages are changes from that "
            "baseline; negative is better for error, high-frequency motion, jerk, acceleration, "
            "and acceleration-limit occupancy."
        ),
        "",
        "## Controller configurations",
        "",
    ]
    actuator_ids = sorted({actuator_id for run in runs for actuator_id in run.gains})
    include_low_acceleration_cost = any(run.low_acceleration_cost is not None for run in runs)
    optional_header = " | Low-acceleration cost" if include_low_acceleration_cost else ""
    optional_alignment = "---:|" if include_low_acceleration_cost else ""
    lines.extend(
        [
            "| Run | "
            + " | ".join(f"Actuator {actuator_id}" for actuator_id in actuator_ids)
            + optional_header
            + " |",
            "|---|" + "---:|" * len(actuator_ids) + optional_alignment,
        ]
    )
    for run in runs:
        gain_cells = [
            f"{run.gains[actuator_id].p}/{run.gains[actuator_id].i}/{run.gains[actuator_id].d}"
            for actuator_id in actuator_ids
        ]
        optional_cell = (
            f" | {run.low_acceleration_cost:g}"
            if run.low_acceleration_cost is not None
            else (" | —" if include_low_acceleration_cost else "")
        )
        lines.append(f"| {run.label} | " + " | ".join(gain_cells) + optional_cell + " |")
    lines.extend(["", "Values are shown as `P/I/D`.", ""])

    lines.extend(["## Run artifacts", "", "| Run | Artifacts | Metrics |", "|---|---|---|"])
    for run in runs:
        directory = run.run_directory.relative_to(path.parent)
        metrics_path = run.metrics_path.relative_to(path.parent)
        lines.append(
            f"| {run.label} | [directory]({directory.as_posix()}/) | "
            f"[metrics]({metrics_path.as_posix()}) |"
        )

    for phase, title in PHASES:
        if not any(_mapping_at(payload, "tracking", phase) for payload in metrics):
            continue
        lines.extend(["", f"## {title}", ""])
        _append_joint_metric_table(
            lines,
            runs,
            metrics,
            phase,
            JointMetricSpec(
                title="Joint tracking RMS",
                metric="rms",
                unit="rad",
                branch="tracking",
            ),
        )
        _append_joint_metric_table(
            lines,
            runs,
            metrics,
            phase,
            JointMetricSpec(
                title="High-frequency joint motion RMS",
                metric="high_frequency_rms_rad",
                unit="rad",
                branch="smoothness",
            ),
        )
        _append_joint_metric_table(
            lines,
            runs,
            metrics,
            phase,
            JointMetricSpec(
                title="Measured joint jerk RMS",
                metric="jerk_rms_rad_s3",
                unit="rad/s³",
                branch="smoothness",
            ),
        )
        _append_joint_metric_table(
            lines,
            runs,
            metrics,
            phase,
            JointMetricSpec(
                title="Measured joint acceleration p95",
                metric="acceleration_p95_rad_s2",
                unit="rad/s²",
                branch="smoothness",
            ),
        )
        _append_joint_metric_table(
            lines,
            runs,
            metrics,
            phase,
            JointMetricSpec(
                title="Commanded joint acceleration RMS",
                metric="command_acceleration_rms_rad_s2",
                unit="rad/s²",
                branch="smoothness",
            ),
        )
        _append_joint_metric_table(
            lines,
            runs,
            metrics,
            phase,
            JointMetricSpec(
                title="Commanded joint acceleration p95",
                metric="command_acceleration_p95_rad_s2",
                unit="rad/s²",
                branch="smoothness",
            ),
        )
        _append_joint_metric_table(
            lines,
            runs,
            metrics,
            phase,
            JointMetricSpec(
                title="Joint acceleration-limit occupancy",
                metric="acceleration_limit_fraction",
                unit="fraction",
                branch="smoothness",
            ),
        )
        _append_tool_metric_table(lines, runs, metrics, phase)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_joint_metric_table(
    lines: list[str],
    runs: Sequence[CompletedRun],
    metrics: Sequence[Mapping[str, object]],
    phase: str,
    spec: JointMetricSpec,
) -> None:
    joint_maps = [_mapping_at(payload, spec.branch, phase, "joints") for payload in metrics]
    joint_names = sorted({str(name) for joint_map in joint_maps for name in joint_map})
    if not joint_names or not any(
        _numeric_at(joint_map, joint_name, spec.metric) is not None
        for joint_map in joint_maps
        for joint_name in joint_names
    ):
        return
    lines.extend(
        [
            f"### {spec.title} ({spec.unit})",
            "",
            "| Joint | " + " | ".join(run.label for run in runs) + " |",
            "|---|" + "---:|" * len(runs),
        ]
    )
    for joint_name in joint_names:
        values = [_numeric_at(joint_map, joint_name, spec.metric) for joint_map in joint_maps]
        lines.append(
            f"| {joint_name} | "
            + " | ".join(_comparison_cell(value, values[0]) for value in values)
            + " |"
        )
    lines.append("")


def _append_tool_metric_table(
    lines: list[str],
    runs: Sequence[CompletedRun],
    metrics: Sequence[Mapping[str, object]],
    phase: str,
) -> None:
    rows = (
        (
            "OSC translation tracking RMS",
            "mm",
            1_000.0,
            ("tracking", phase, "osc_tool_translation_m", "rms"),
        ),
        (
            "OSC orientation tracking RMS",
            "rad",
            1.0,
            ("tracking", phase, "osc_tool_orientation_rad", "rms"),
        ),
        (
            "End-to-end translation tracking RMS",
            "mm",
            1_000.0,
            ("tracking", phase, "end_to_end_tool_translation_m", "rms"),
        ),
        (
            "End-to-end orientation tracking RMS",
            "rad",
            1.0,
            ("tracking", phase, "end_to_end_tool_orientation_rad", "rms"),
        ),
        (
            "High-frequency tool motion RMS",
            "mm",
            1_000.0,
            ("smoothness", phase, "tool_high_frequency_rms_m"),
        ),
    )
    if not any(_numeric_at(payload, *keys) is not None for payload in metrics for *_, keys in rows):
        return
    lines.extend(
        [
            "### Tool tracking and smoothness",
            "",
            "| Metric | Unit | " + " | ".join(run.label for run in runs) + " |",
            "|---|---:|" + "---:|" * len(runs),
        ]
    )
    for label, unit, scale, keys in rows:
        values = [
            value * scale if (value := _numeric_at(payload, *keys)) is not None else None
            for payload in metrics
        ]
        lines.append(
            f"| {label} | {unit} | "
            + " | ".join(_comparison_cell(value, values[0]) for value in values)
            + " |"
        )
    lines.append("")


def _mapping_at(value: object, *keys: str) -> Mapping[str, object]:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = cast(Mapping[object, object], current).get(key)
    return cast(Mapping[str, object], current) if isinstance(current, Mapping) else {}


def _numeric_at(value: object, *keys: str) -> float | None:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = cast(Mapping[object, object], current).get(key)
    if isinstance(current, int | float) and not isinstance(current, bool):
        return float(current)
    return None


def _comparison_cell(value: float | None, baseline: float | None) -> str:
    if value is None:
        return "—"
    rendered = f"{value:.5g}"
    if baseline is None or baseline in {0.0, value}:
        return rendered
    change = (value - baseline) / abs(baseline) * 100.0
    return f"{rendered} ({change:+.1f}%)"


def _run_sweep(  # noqa: PLR0915 - coordinates safety-critical sweep and cleanup in one scope
    args: argparse.Namespace,
    forwarded_arguments: Sequence[str],
) -> Path:
    actuator_ids = sorted(POSITION_PID_GAINS_BY_ACTUATOR_ID)
    specs = load_sweep(args.sweep_file, actuator_ids)
    tracking_settings = controller_tracking_settings_from_arguments(forwarded_arguments)
    if not isinstance(args.stack_startup_delay, int | float) or args.stack_startup_delay < 0.0:
        raise ValueError("Stack startup delay must be non-negative.")
    _confirm_hardware_sweep(specs, assume_yes=args.yes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_directory = args.output_dir / f"controller_sweep_{timestamp}"
    sweep_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = sweep_directory / "manifest.json"
    comparison_path = sweep_directory / "comparison.md"
    dashboard_state: DashboardState | None = None
    original_gains: dict[int, FeetechPIDGains] | None = None
    completed_runs: list[CompletedRun] = []
    run_records: list[dict[str, object]] = []
    failure: Exception | None = None
    restore_error: Exception | None = None

    _write_manifest(
        manifest_path,
        sweep_file=args.sweep_file,
        original_gains=None,
        run_records=run_records,
        status="preparing",
    )
    try:
        dashboard_state = _prepare_dashboard(args.dashboard_url)
        _assert_no_external_stack()
        logger.info("Reading the PID gains currently stored on the robot")
        original_gains = _read_current_gains(actuator_ids)
        resolved_runs = [(spec, spec.resolve(original_gains)) for spec in specs]
        _write_manifest(
            manifest_path,
            sweep_file=args.sweep_file,
            original_gains=original_gains,
            run_records=run_records,
            status="running",
        )

        for index, (spec, gains) in enumerate(resolved_runs, start=1):
            run_directory = sweep_directory / spec.label
            run_directory.mkdir()
            robot_config = resolve_robot_config(
                TRISKEL_CONFIG,
                gains,
                low_acceleration_cost=spec.low_acceleration_cost,
            )
            resolved_osc = robot_config.operational_space_config
            resolved_low_acceleration_cost = (
                resolved_osc.low_acceleration_cost if resolved_osc is not None else None
            )
            logger.info(
                "Controller sweep run %d/%d: %s",
                index,
                len(resolved_runs),
                spec.label,
            )
            record: dict[str, object] = {
                "label": spec.label,
                "status": "running",
                "gains": _gains_json(gains),
                "low_acceleration_cost": resolved_low_acceleration_cost,
                "robot_config": robot_config,
                "directory": spec.label,
                "diagnostic_arguments": list(forwarded_arguments),
            }
            run_records.append(record)
            _write_manifest(
                manifest_path,
                sweep_file=args.sweep_file,
                original_gains=original_gains,
                run_records=run_records,
                status="running",
            )
            try:
                _run_with_config(
                    robot_config,
                    diagnostic=DiagnosticInvocation(
                        settings=tracking_settings,
                        label=spec.label,
                        output_directory=run_directory,
                    ),
                    startup_delay_s=float(args.stack_startup_delay),
                )
                _require_completed_diagnostic(run_directory)
                metrics_path = _find_metrics(run_directory)
            except Exception as exc:
                diagnostic_failure = RuntimeError(
                    _diagnostic_failure_reason(run_directory) or str(exc)
                )
                record["status"] = "failed"
                record["error"] = str(diagnostic_failure)
                raise diagnostic_failure from exc
            record["status"] = "completed"
            record["metrics"] = str(metrics_path.relative_to(sweep_directory))
            completed_runs.append(
                CompletedRun(
                    label=spec.label,
                    gains=gains,
                    metrics_path=metrics_path,
                    run_directory=run_directory,
                    low_acceleration_cost=resolved_low_acceleration_cost,
                )
            )
            write_sweep_report(comparison_path, completed_runs)
    except Exception as exc:
        failure = exc
    finally:
        if original_gains is not None:
            try:
                logger.info("Restoring the PID gains present before the sweep")
                restore_config = resolve_robot_config(
                    TRISKEL_CONFIG,
                    original_gains,
                    low_acceleration_cost=None,
                )
                _run_with_config(
                    restore_config,
                    diagnostic=None,
                    startup_delay_s=0.0,
                )
            except Exception as exc:
                restore_error = exc
                logger.exception("Could not restore the original hardware PID gains")
        try:
            _restore_dashboard(dashboard_state)
        except Exception as exc:
            if restore_error is None:
                restore_error = exc
            logger.exception("Could not restore the dashboard stack")

    final_error = restore_error or failure
    status = "completed" if final_error is None else "failed"
    _write_manifest(
        manifest_path,
        sweep_file=args.sweep_file,
        original_gains=original_gains,
        run_records=run_records,
        status=status,
        error=str(final_error) if final_error is not None else None,
    )
    if completed_runs:
        write_sweep_report(comparison_path, completed_runs)
    if restore_error is not None:
        raise RuntimeError(
            f"Controller sweep cleanup failed; verify the robot gains before use: {restore_error}"
        ) from restore_error
    if failure is not None:
        raise RuntimeError(str(failure)) from failure
    return comparison_path


def main() -> None:
    setup_logging()
    parser = _build_parser()
    args, forwarded_arguments = parser.parse_known_args()
    try:
        forwarded_arguments = _validate_tracking_arguments(forwarded_arguments)
        comparison_path = _run_sweep(args, forwarded_arguments)
    except (RuntimeError, ValueError) as exc:
        logger.error("Controller sweep failed: %s", exc)
        sys.exit(1)
    logger.info("Controller sweep comparison: %s", comparison_path)


if __name__ == "__main__":
    main()
