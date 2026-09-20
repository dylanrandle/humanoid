"""Frozen Triskel simulation experiment and predeclared test scenarios."""

import math
from dataclasses import replace

from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.osc_tuning import OSCTuningGates, OSCTuningScenario, TuningSuite

BASELINE_OSC_CONFIG = OperationalSpaceConfig(
    dt=1 / 30,
    tool_position_cost=5.0,
    tool_orientation_cost=1.0,
    avoid_collisions=True,
    min_collision_distance=0.005,
    collision_safe_displacement_gain=0.01,
    manipulability_cost=0.005,
    damping_cost=0.1,
    low_acceleration_cost=0.03,
    joint_velocity_limit=1.0,
    joint_acceleration_limit=2.0,
)
TUNING_GATES = OSCTuningGates(
    position_p95_m=0.005,
    position_max_m=0.015,
    orientation_p95_rad=math.radians(2),
    orientation_max_rad=math.radians(5),
    settling_s=1.0,
    clearance_tolerance_m=1e-6,
    limit_tolerance=1e-6,
    tool_smoothness_scale_m=0.001,
    joint_smoothness_scale_rad=math.radians(0.25),
)
TRAINING_ANCHORS = {
    "home": (0.0, -0.75, 0.5, 0.0, 0.0, 1.0, 0.0),
    "offset": (0.2, -0.9, 0.65, 0.15, -0.15, 0.9, 0.15),
}
VALIDATION_ANCHORS = {
    "unseen": (-0.2, -0.85, 0.6, -0.12, 0.18, 1.1, -0.15),
    "extended": (0.1, -0.5, 0.2, 0.1, -0.1, 0.45, 0.1),
    "folded": (-0.1, -0.95, 0.85, 0.25, 0.1, 1.0, -0.1),
}
CONFIRMATION_ANCHORS = {
    "confirmation_a": (0.12, -0.82, 0.58, -0.08, -0.12, 1.02, 0.12),
    "confirmation_b": (-0.16, -0.58, 0.28, -0.10, 0.12, 0.53, -0.08),
}


def tuning_scenarios(suite: TuningSuite) -> tuple[OSCTuningScenario, ...]:
    """Return the fixed suite; validation scenarios never participate in screening."""
    if suite == "confirmation":
        return _confirmation_scenarios()
    if suite in {"training", "screen"}:
        scenarios = []
        for name, arm in TRAINING_ANCHORS.items():
            for orientation in (0.0, 10.0):
                for plane in ("xy", "xz", "yz"):
                    scenarios.extend(
                        OSCTuningScenario(
                            name=f"{name}_{plane}_{scale:g}x_{orientation:g}deg",
                            arm_positions=arm,
                            plane=plane,
                            scale=scale,
                            orientation_deg=orientation,
                        )
                        for scale in (1.0, 1.5, 2.0)
                    )
            scenarios.extend(
                OSCTuningScenario(
                    name=f"{name}_{motion}",
                    arm_positions=arm,
                    motion=motion,
                )
                for motion in ("translation_ramp", "orientation_ramp", "hold")
            )
        if suite == "screen":
            names = {
                "home_xy_2x_10deg",
                "home_xz_2x_0deg",
                "offset_yz_2x_10deg",
                "home_translation_ramp",
                "offset_orientation_ramp",
                "offset_hold",
            }
            scenarios = [scenario for scenario in scenarios if scenario.name in names]
        return tuple(scenarios)
    if suite == "robustness":
        return (
            OSCTuningScenario(
                name="unreachable_then_recovery",
                arm_positions=TRAINING_ANCHORS["home"],
                motion="unreachable",
                hold_s=10.0,
            ),
        )
    scenarios = [
        OSCTuningScenario(
            name=f"{name}_{plane}_6s",
            arm_positions=arm,
            plane=plane,
            period_s=6.0,
            scale=1.0,
        )
        for name, arm in VALIDATION_ANCHORS.items()
        for plane in ("xy", "xz", "yz")
    ]
    unseen = VALIDATION_ANCHORS["unseen"]
    for motion in ("translation_ramp", "orientation_ramp", "hold", "base"):
        scenarios.append(
            OSCTuningScenario(
                name=f"unseen_{motion}",
                arm_positions=unseen,
                motion=motion,
            )
        )
    # Sensitivity cases are assumptions, not identified Triskel hardware uncertainty.
    reference = scenarios[0]
    scenarios.extend(
        (
            replace(reference, name="unseen_payload_100g", payload_kg=0.1),
            replace(reference, name="unseen_damping_half", damping_multiplier=0.5),
            replace(reference, name="unseen_damping_double", damping_multiplier=2.0),
            replace(reference, name="unseen_reference_delay", delay_s=1 / 30),
            replace(
                reference,
                name="unseen_reference_jitter",
                reference_jitter_m=0.0005,
                reference_jitter_rad=math.radians(0.25),
            ),
        )
    )
    return tuple(scenarios)


def _confirmation_scenarios() -> tuple[OSCTuningScenario, ...]:
    """Fresh poses/speed reserved after real-time latency required cost refinement."""
    scenarios = [
        OSCTuningScenario(
            name=f"{name}_{plane}_7s",
            arm_positions=arm,
            plane=plane,
            period_s=7.0,
            scale=1.5,
        )
        for name, arm in CONFIRMATION_ANCHORS.items()
        for plane in ("xy", "xz", "yz")
    ]
    scenarios.extend(
        OSCTuningScenario(
            name=f"confirmation_a_{motion}",
            arm_positions=CONFIRMATION_ANCHORS["confirmation_a"],
            motion=motion,
        )
        for motion in ("translation_ramp", "orientation_ramp", "hold", "base")
    )
    reference = scenarios[0]
    scenarios.extend(
        (
            replace(reference, name="confirmation_a_payload", payload_kg=0.1),
            replace(reference, name="confirmation_a_delay", delay_s=1 / 30),
            replace(
                reference,
                name="confirmation_a_jitter",
                reference_jitter_m=0.0005,
                reference_jitter_rad=math.radians(0.25),
            ),
        )
    )
    return tuple(scenarios)
