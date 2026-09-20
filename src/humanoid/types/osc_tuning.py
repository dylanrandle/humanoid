"""Shared experiment definitions for offline OSC tuning."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TuningMotion = Literal[
    "figure_eight", "translation_ramp", "orientation_ramp", "hold", "base", "unreachable"
]
TuningSuite = Literal["screen", "training", "validation", "confirmation", "robustness"]
TuningTrace = dict[str, NDArray[np.float64]]


@dataclass(frozen=True, kw_only=True)
class OSCTuningScenario:
    """One reset, reference trajectory, and explicitly assumed model perturbation."""

    name: str
    arm_positions: tuple[float, ...]
    motion: TuningMotion = "figure_eight"
    plane: Literal["xy", "xz", "yz"] = "xy"
    scale: float = 1.0
    period_s: float = 8.0
    orientation_deg: float = 10.0
    hold_s: float = 3.0
    delay_s: float = 0.0
    payload_kg: float = 0.0
    damping_multiplier: float = 1.0
    reference_jitter_m: float = 0.0
    reference_jitter_rad: float = 0.0

    @property
    def motion_duration_s(self) -> float:
        if self.motion == "figure_eight":
            return 2.25 * self.period_s
        if self.motion in {"translation_ramp", "orientation_ramp"}:
            return 1.0
        return 10.0

    @property
    def duration_s(self) -> float:
        return self.motion_duration_s + self.hold_s


@dataclass(frozen=True, kw_only=True)
class OSCTuningGates:
    """Experiment thresholds, independent of the candidate's cost weights."""

    position_p95_m: float
    position_max_m: float
    orientation_p95_rad: float
    orientation_max_rad: float
    settling_s: float
    clearance_tolerance_m: float
    limit_tolerance: float
    tool_smoothness_scale_m: float
    joint_smoothness_scale_rad: float
