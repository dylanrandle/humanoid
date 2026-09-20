"""Configuration and result types for controller-tracking sweeps."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from humanoid.hardware.actuators.feetech.config import FeetechPIDGains
from humanoid.robots.utils.controller_tracking.models import ControllerTrackingSettings


@dataclass(frozen=True)
class SweepRunSpec:
    """One named controller configuration from the user-provided sweep file."""

    label: str
    default: FeetechPIDGains | None
    overrides: Mapping[int, FeetechPIDGains]
    low_acceleration_cost: float | None = None

    def resolve(
        self,
        baseline: Mapping[int, FeetechPIDGains],
    ) -> dict[int, FeetechPIDGains]:
        resolved = (
            dict.fromkeys(baseline, self.default) if self.default is not None else dict(baseline)
        )
        resolved.update(self.overrides)
        return resolved


@dataclass(frozen=True)
class CompletedRun:
    """Artifacts retained for one successful diagnostic invocation."""

    label: str
    gains: Mapping[int, FeetechPIDGains]
    metrics_path: Path
    run_directory: Path
    low_acceleration_cost: float | None = None


@dataclass(frozen=True)
class DashboardState:
    """Dashboard configuration that should be restored after the sweep."""

    base_url: str
    stack_was_running: bool
    runtime: str
    robot: str


@dataclass(frozen=True)
class DiagnosticInvocation:
    """One in-process diagnostic using the same config as the managed stack."""

    settings: ControllerTrackingSettings
    label: str
    output_directory: Path


@dataclass(frozen=True)
class JointMetricSpec:
    """One joint-level metric rendered in every available tracking phase."""

    title: str
    metric: str
    unit: str
    branch: str
