"""Reproducibility metadata and A/B comparison reports."""

import json
import subprocess
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from humanoid.robots.utils.controller_tracking.models import (
    ControllerTrackingSettings,
    TrackingRun,
)
from humanoid.types.robot import RobotConfig
from humanoid.utils.paths import find_data_root


def write_run_metadata(
    path: Path,
    settings: ControllerTrackingSettings,
    robot_config: RobotConfig,
    *,
    label: str | None,
    run: TrackingRun,
) -> None:
    """Save the configuration and source revision needed to interpret a run."""
    hardware_actuators = (
        robot_config.hardware.actuators if robot_config.hardware is not None else None
    )
    payload = {
        "label": label,
        "robot": robot_config.name.value,
        "completed": run.completed,
        "failure_reason": run.failure_reason,
        "settings": asdict(settings),
        "operational_space_config": robot_config.operational_space_config,
        "actuators": (hardware_actuators.joints if hardware_actuators is not None else {}),
        "git": _git_metadata(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_run_comparison(
    path: Path,
    current_metrics_path: Path,
    baseline: Path,
) -> Path:
    """Compare matching scalar tracking and smoothness metrics with a prior run."""
    baseline_metrics_path = _resolve_metrics_path(baseline, exclude=current_metrics_path)
    current = json.loads(current_metrics_path.read_text(encoding="utf-8"))
    previous = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    current_values = _flatten_numeric_values(current)
    previous_values = _flatten_numeric_values(previous)
    shared_paths = sorted(
        metric_path
        for metric_path in current_values.keys() & previous_values.keys()
        if not metric_path.endswith(("dominant_frequency_hz", "shake_cutoff_hz"))
    )
    lines = [
        "# Controller tracking A/B comparison",
        "",
        f"Baseline: `{baseline_metrics_path}`",
        "",
        "Negative change is an improvement for every error and smoothness metric below.",
        "",
        "| Metric | Baseline | Current | Change |",
        "|---|---:|---:|---:|",
    ]
    for metric_path in shared_paths:
        baseline_value = previous_values[metric_path]
        current_value = current_values[metric_path]
        if baseline_value == 0.0:
            change = "—"
        else:
            change = f"{(current_value - baseline_value) / abs(baseline_value) * 100.0:+.1f}%"
        lines.append(f"| `{metric_path}` | {baseline_value:.6g} | {current_value:.6g} | {change} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return baseline_metrics_path


def _resolve_metrics_path(path: Path, *, exclude: Path | None = None) -> Path:
    excluded = exclude.resolve() if exclude is not None else None
    if path.is_file() and path.resolve() != excluded:
        return path
    if path.is_dir():
        candidates = sorted(
            (
                candidate
                for candidate in path.glob("*_metrics.json")
                if candidate.resolve() != excluded
            ),
            key=lambda item: item.stat().st_mtime,
        )
        if candidates:
            return candidates[-1]
    raise ValueError(f"No metrics JSON found at {path}")


def _flatten_numeric_values(value: object, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        flattened = {}
        for key, nested in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric_values(nested, nested_prefix))
        return flattened
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def _git_metadata() -> dict[str, str | bool | None]:
    root = find_data_root(__file__)
    revision = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain")
    return {
        "revision": revision,
        "dirty": None if status is None else bool(status),
    }


def _run_git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _json_ready(value: Any) -> Any:  # noqa: PLR0911 - recursive serializer boundary
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value
