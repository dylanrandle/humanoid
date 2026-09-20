"""Resettable, network-free Triskel OSC experiments in native MuJoCo.

Example: uv run python -m humanoid.robots.utils.osc_tuning --name baseline --suite screen
This module never constructs hardware drivers, publishers, or subscribers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import pink
import pinocchio as pin

from humanoid.config.osc_tuning import BASELINE_OSC_CONFIG, TUNING_GATES, tuning_scenarios
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.config.simulation import DEFAULT_MUJOCO_SIMULATION_CONFIG
from humanoid.controllers.omniwheel_base import OmniwheelBaseController
from humanoid.controllers.operational_space import OperationalSpaceController
from humanoid.robots.utils.controller_tracking.models import (
    ControllerTrackingSettings,
    FigureEightSetting,
)
from humanoid.robots.utils.controller_tracking.smoothness import analyze_smoothness
from humanoid.robots.utils.controller_tracking.trajectory import (
    figure_eight_pose,
    figure_eight_velocity,
)
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.controller_tracking import NativeJointSample
from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.homing import HomingPreset
from humanoid.types.osc_tuning import OSCTuningScenario, TuningTrace
from humanoid.types.robot import CartesianVelocity, RobotJointCommand
from humanoid.utils.paths import find_repo_root

SETUP_SETTLE_S = 2.0
JACOBIAN_LENGTH_SCALE_M = 0.25


def reference_at(
    scenario: OSCTuningScenario, anchor: pin.SE3, elapsed_s: float
) -> tuple[pin.SE3, CartesianVelocity]:
    """Apply an optional, explicitly assumed 6 Hz perturbation to the reference."""
    pose, velocity = _reference_at(scenario, anchor, elapsed_s)
    if not scenario.reference_jitter_m and not scenario.reference_jitter_rad:
        return pose, velocity
    t = float(np.clip(elapsed_s, 0, scenario.motion_duration_s))
    frequency_rad_s = 2 * np.pi * 6
    envelope = np.sin(np.pi * t / scenario.motion_duration_s) ** 2
    envelope_rate = (
        np.pi / scenario.motion_duration_s * np.sin(2 * np.pi * t / scenario.motion_duration_s)
    )
    offset = envelope * np.sin(frequency_rad_s * t)
    rate = envelope_rate * np.sin(frequency_rad_s * t) + envelope * frequency_rad_s * np.cos(
        frequency_rad_s * t
    )
    pose.translation[0] += scenario.reference_jitter_m * offset
    rotation = pin.exp3(np.array([0.0, 0.0, scenario.reference_jitter_rad * offset]))
    pose.rotation = rotation @ pose.rotation
    return pose, CartesianVelocity(
        linear=velocity.linear + np.array([scenario.reference_jitter_m * rate, 0.0, 0.0]),
        angular=rotation @ velocity.angular
        + np.array([0.0, 0.0, scenario.reference_jitter_rad * rate]),
    )


def _reference_at(
    scenario: OSCTuningScenario, anchor: pin.SE3, elapsed_s: float
) -> tuple[pin.SE3, CartesianVelocity]:
    """Continuous reference with analytic feedforward, including a final hold."""
    zero = np.zeros(3)
    if scenario.motion == "figure_eight":
        settings = ControllerTrackingSettings(
            period_s=scenario.period_s, orientation_bias_rad=np.deg2rad(scenario.orientation_deg)
        )
        setting = FigureEightSetting(plane=scenario.plane, size_multiplier=scenario.scale)
        return (
            figure_eight_pose(anchor, elapsed_s, settings, setting),
            figure_eight_velocity(anchor, elapsed_s, settings, setting),
        )
    target = anchor.copy()
    u = float(np.clip(elapsed_s / scenario.motion_duration_s, 0, 1))
    blend = u**3 * (10 - 15 * u + 6 * u**2)
    rate = 30 * u**2 * (1 - u) ** 2 / scenario.motion_duration_s
    if scenario.motion == "translation_ramp":
        displacement = np.array([0.01, 0.0, 0.0])
        target.translation += blend * displacement
        return target, CartesianVelocity(linear=rate * displacement, angular=zero)
    if scenario.motion == "orientation_ramp":
        rotation = np.array([0.0, 0.0, np.deg2rad(5)])
        target.rotation = pin.exp3(blend * rotation) @ target.rotation
        return target, CartesianVelocity(linear=zero, angular=rate * rotation)
    if scenario.motion == "unreachable" and elapsed_s < scenario.motion_duration_s:
        target.translation[2] += 1.0
    return target, CartesianVelocity(linear=zero, angular=zero)


def base_reference(elapsed_s: float) -> pin.SE3:
    """Slow 0.1 m translation and 15 degree yaw over ten seconds, then hold."""
    u = float(np.clip(elapsed_s / 10, 0, 1))
    blend = u**3 * (10 - 15 * u + 6 * u**2)
    return pin.SE3(
        pin.exp3(np.array([0.0, 0.0, np.deg2rad(15) * blend])), np.array([0.1 * blend, 0.0, 0.0])
    )


def _configuration(engine: NativeMujocoEngine, q: np.ndarray) -> pink.Configuration:
    robot = engine.robot
    return pink.Configuration(
        robot.model,
        robot.model.createData(),
        q,
        collision_model=robot.collision_model,
        collision_data=pin.GeometryData(robot.collision_model),
    )


def _clearance(configuration: pink.Configuration) -> float:
    return min(result.min_distance for result in configuration.collision_data.distanceResults)


def _dexterity(configuration: pink.Configuration, indices: np.ndarray, frame: str) -> np.ndarray:
    jacobian = configuration.get_frame_jacobian(frame)[:, indices]
    log_volume = 0.5 * np.linalg.slogdet(jacobian @ jacobian.T + 1e-6 * np.eye(6))[1]
    scaled = jacobian.copy()
    scaled[:3] /= JACOBIAN_LENGTH_SCALE_M
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    return np.array([log_volume, singular_values[-1], np.prod(singular_values)])


def _native_tool_pose(engine: NativeMujocoEngine) -> pin.SE3:
    """Ground truth from MuJoCo bodies, in the physical base's frame."""
    mujoco.mj_forward(engine.model, engine.data)
    body = engine.model.body(engine.robot.config.tool.frame).id
    world_tool = pin.SE3(engine.data.xmat[body].reshape(3, 3), engine.data.xpos[body])
    root = engine.binding.root
    assert root is not None
    x, y, yaw = engine.data.qpos[list(root.qpos_addresses)]
    world_base = pin.SE3(pin.exp3(np.array([0.0, 0.0, yaw])), np.array([x, y, 0.0]))
    return world_base.actInv(world_tool)


def _make_engine(scenario: OSCTuningScenario) -> NativeMujocoEngine:
    home = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy()
    # Resolve nq indices from the arm chain rather than assuming nq == nv.
    engine = NativeMujocoEngine(TRISKEL_CONFIG)
    arm = engine.robot.get_arm_joint_indices()
    home[engine.robot.get_joint_position_indices(arm)] = scenario.arm_positions
    if scenario.payload_kg:
        body = engine.model.body(engine.robot.config.tool.frame).id
        # A compact payload at the existing tool-body COM; assumes a 2 cm radius.
        engine.model.body_mass[body] += scenario.payload_kg
        engine.model.body_inertia[body] += 0.4 * scenario.payload_kg * 0.02**2
        mujoco.mj_setConst(engine.model, engine.data)
    engine.model.dof_damping[:] *= scenario.damping_multiplier
    engine.reset(initial_q=home)
    return engine


def run_scenario(  # noqa: PLR0915 - keep the simulation event order visible
    config: OperationalSpaceConfig, scenario: OSCTuningScenario
) -> tuple[TuningTrace, dict[str, object]]:
    """Integrate commanded state as the active controller node does, without feedback resets."""
    engine = _make_engine(scenario)
    osc = OperationalSpaceController(engine.robot, config)
    if not config.avoid_collisions:
        raise ValueError("Tuning requires collision checking to remain enabled.")
    engine.step(round(SETUP_SETTLE_S / engine.physics_timestep))
    initial = engine.read_robot_state(0.0).joint_positions
    osc.update_state(initial)
    anchor = engine.robot.get_tool_command_pose(initial).copy()
    base = OmniwheelBaseController(engine.robot, TRISKEL_CONFIG.omniwheel_base_config)
    base.update_state(initial)
    measured = _configuration(engine, initial)
    commanded = _configuration(engine, initial)
    q_command = initial.copy()
    v_command = np.zeros(engine.robot.model.nv)
    samples: dict[str, list] = {
        name: []
        for name in (
            "time",
            "q",
            "v",
            "reference_position",
            "reference_rotation",
            "tool_position",
            "position_error",
            "orientation_error",
            "clearance",
            "native_fk_error",
            "native_orientation_error",
            "command_time",
            "dt",
            "command_q",
            "command_v",
            "command_tool_position",
            "command_position_error",
            "command_orientation_error",
            "command_clearance",
            "dexterity",
            "solve_duration",
            "solve_success",
        )
    }
    errors = []
    tick = 0
    last_control_s = 0.0
    started_s = time.perf_counter()
    steps = round(scenario.duration_s / engine.physics_timestep)
    for step in range(steps + 1):
        elapsed = step * engine.physics_timestep
        reference, _ = reference_at(scenario, anchor, elapsed)
        if elapsed + 1e-10 >= tick * config.dt and step < steps:
            dt = config.dt if tick == 0 else elapsed - last_control_s
            target, velocity = reference_at(scenario, anchor, max(0.0, elapsed - scenario.delay_s))
            result = osc.compute_control(target, dt=dt, target_velocity=velocity)
            q_command, v_command = result.q, result.v
            if scenario.motion == "base":
                base_result = base.compute_control(base_reference(elapsed), dt=dt)
                q_command[base.controlled_q_indices] = base_result.q[base.controlled_q_indices]
                v_command[base.controlled_v_indices] = base_result.v[base.controlled_v_indices]
            osc.update_state(q_command)
            base.update_state(q_command)
            engine.apply_joint_command(
                RobotJointCommand(
                    timestamp=elapsed,
                    joint_positions=q_command,
                    joint_velocities=v_command,
                )
            )
            commanded.update(q_command)
            command_pose = engine.robot.get_tool_command_pose(q_command).copy()
            assert result.diagnostics is not None
            if not result.diagnostics.succeeded:
                errors.append({"time": elapsed, "error": result.diagnostics.error})
            command_values = {
                "command_time": elapsed,
                "dt": dt,
                "command_q": q_command.copy(),
                "command_v": v_command.copy(),
                "command_tool_position": command_pose.translation.copy(),
                "command_position_error": np.linalg.norm(
                    command_pose.translation - reference.translation
                ),
                "command_orientation_error": np.linalg.norm(
                    pin.log3(reference.rotation.T @ command_pose.rotation)
                ),
                "command_clearance": _clearance(commanded),
                "dexterity": _dexterity(
                    commanded, osc.controlled_v_indices, engine.robot.config.tool.frame
                ),
                "solve_duration": result.diagnostics.duration_s,
                "solve_success": float(result.diagnostics.succeeded),
            }
            for name, value in command_values.items():
                samples[name].append(value)
            tick += 1
            last_control_s = elapsed
        state = engine.read_robot_state(elapsed)
        measured.update(state.joint_positions)
        pose = engine.robot.get_tool_command_pose(state.joint_positions).copy()
        native_pose = _native_tool_pose(engine)
        values = {
            "time": elapsed,
            "q": state.joint_positions,
            "v": state.joint_velocities,
            "reference_position": reference.translation.copy(),
            "reference_rotation": reference.rotation.copy(),
            "tool_position": pose.translation.copy(),
            "position_error": np.linalg.norm(pose.translation - reference.translation),
            "orientation_error": np.linalg.norm(pin.log3(reference.rotation.T @ pose.rotation)),
            "clearance": _clearance(measured),
            "native_fk_error": np.linalg.norm(pose.translation - native_pose.translation),
            "native_orientation_error": np.linalg.norm(
                pin.log3(pose.rotation.T @ native_pose.rotation)
            ),
        }
        for name, value in values.items():
            samples[name].append(value)
        if step < steps:
            engine.step()
    trace = {name: np.asarray(value, dtype=float) for name, value in samples.items()}
    metadata = {
        "initial_q": initial.tolist(),
        "initial_anchor": anchor.homogeneous.tolist(),
        "wall_duration_s": time.perf_counter() - started_s,
        "solver_errors": errors,
        "mujoco_warnings": engine.data.warning.number.tolist(),
        "arm_q_indices": osc.controlled_q_indices.tolist(),
        "arm_v_indices": osc.controlled_v_indices.tolist(),
        "arm_names": [engine.robot.joint_idx_to_name(i) for i in osc.controlled_joint_indices],
        "lower_limits": engine.robot.model.lowerPositionLimit[osc.controlled_q_indices].tolist(),
        "upper_limits": engine.robot.model.upperPositionLimit[osc.controlled_q_indices].tolist(),
        "urdf_velocity_limits": engine.robot.model.velocityLimit[osc.controlled_v_indices].tolist(),
    }
    return trace, metadata


def analyze_trial(
    trace: TuningTrace, metadata: dict, config: OperationalSpaceConfig, scenario: OSCTuningScenario
) -> dict[str, object]:
    """Apply gates before ranking; raw velocities check hard acceleration limits."""
    gates = TUNING_GATES
    q_indices, v_indices = metadata["arm_q_indices"], metadata["arm_v_indices"]
    native_samples = []
    for stream, prefix in (("state", ""), ("controller", "command_")):
        for index, timestamp in enumerate(trace[f"{prefix}time"]):
            native_samples.append(
                NativeJointSample(
                    segment="osc_tuning",
                    setting=scenario.name,
                    window_index=0,
                    stream=stream,
                    received_timestamp_s=timestamp,
                    source_timestamp_s=timestamp,
                    joint_names=tuple(metadata["arm_names"]),
                    joint_positions_rad=trace[f"{prefix}q"][index, q_indices],
                    joint_velocities_rad_s=trace[f"{prefix}v"][index, v_indices],
                    tool_position_m=trace[f"{prefix}tool_position"][index],
                )
            )
    smoothness = analyze_smoothness(
        native_samples,
        motion_segments=("osc_tuning",),
        settings=ControllerTrackingSettings(),
    )
    assert smoothness is not None and smoothness.tool_high_frequency_rms_m is not None
    joint_hf = max(stat.high_frequency_rms_rad for stat in smoothness.statistics)
    jerk = max(
        float(np.sqrt(np.mean(np.square(t.jerks_rad_s3), axis=0)).max())
        for t in smoothness.command_traces
    )
    command_v = trace["command_v"][:, v_indices]
    acceleration = (
        np.diff(command_v, axis=0, prepend=np.zeros((1, len(v_indices)))) / trace["dt"][:, None]
    )
    max_acceleration = float(np.abs(acceleration).max())
    velocity_limit = np.minimum(
        metadata["urdf_velocity_limits"],
        np.inf if config.joint_velocity_limit is None else config.joint_velocity_limit,
    )
    acceleration_limit = (
        np.inf if config.joint_acceleration_limit is None else config.joint_acceleration_limit
    )
    margin = min(
        float(
            np.minimum(
                trace[key][:, q_indices] - metadata["lower_limits"],
                metadata["upper_limits"] - trace[key][:, q_indices],
            ).min()
        )
        for key in ("q", "command_q")
    )
    position_p95 = float(np.percentile(trace["position_error"], 95))
    position_max = float(trace["position_error"].max())
    orientation_p95 = float(np.percentile(trace["orientation_error"], 95))
    orientation_max = float(trace["orientation_error"].max())
    clearance = min(float(trace["clearance"].min()), float(trace["command_clearance"].min()))
    nominal = scenario.motion != "unreachable"
    failed = []
    checks = {
        "position_p95": not nominal or position_p95 <= gates.position_p95_m,
        "position_max": not nominal or position_max <= gates.position_max_m,
        "orientation_p95": not nominal or orientation_p95 <= gates.orientation_p95_rad,
        "orientation_max": not nominal or orientation_max <= gates.orientation_max_rad,
        "solver": bool(np.all(trace["solve_success"] == 1)),
        "finite": all(np.isfinite(values).all() for values in trace.values()),
        "position_limits": margin >= -gates.limit_tolerance,
        "velocity_limits": bool(
            np.all(np.abs(command_v) <= velocity_limit + gates.limit_tolerance)
        ),
        "acceleration_limits": bool(
            np.all(np.abs(acceleration) <= acceleration_limit + gates.limit_tolerance)
        ),
        "clearance": clearance >= config.min_collision_distance - gates.clearance_tolerance_m,
        "simulator_warnings": not any(metadata["mujoco_warnings"]),
    }
    settling = None
    if scenario.motion in {"translation_ramp", "orientation_ramp", "unreachable"}:
        within = (trace["position_error"] <= gates.position_p95_m) & (
            trace["orientation_error"] <= gates.orientation_p95_rad
        )
        for index in np.flatnonzero(trace["time"] >= scenario.motion_duration_s):
            end = trace["time"][index] + 1.0
            if end > trace["time"][-1] + 1e-9:
                break
            if np.all(within[index : np.searchsorted(trace["time"], end, side="right")]):
                settling = float(trace["time"][index] - scenario.motion_duration_s)
                break
        checks["settling"] = settling is not None and (not nominal or settling <= gates.settling_s)
    failed.extend(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failed,
        "failed_gates": failed,
        "position_p95_m": position_p95,
        "position_max_m": position_max,
        "orientation_p95_rad": orientation_p95,
        "orientation_max_rad": orientation_max,
        "command_position_p95_m": float(np.percentile(trace["command_position_error"], 95)),
        "command_orientation_p95_rad": float(np.percentile(trace["command_orientation_error"], 95)),
        "tool_hf_rms_m": smoothness.tool_high_frequency_rms_m,
        "worst_joint_hf_rms_rad": joint_hf,
        "worst_command_jerk_rms_rad_s3": jerk,
        "smoothness_score": max(
            smoothness.tool_high_frequency_rms_m / gates.tool_smoothness_scale_m,
            joint_hf / gates.joint_smoothness_scale_rad,
        ),
        "max_command_velocity_rad_s": float(np.abs(command_v).max()),
        "max_command_acceleration_rad_s2": max_acceleration,
        "max_measured_velocity_rad_s": float(np.abs(trace["v"][:, v_indices]).max()),
        "min_joint_margin_rad": margin,
        "min_clearance_m": clearance,
        "solver_failures": int(np.sum(trace["solve_success"] != 1)),
        "solve_p95_s": float(np.percentile(trace["solve_duration"], 95)),
        "native_fk_max_m": float(trace["native_fk_error"].max()),
        "native_fk_max_rad": float(trace["native_orientation_error"].max()),
        "settling_s": settling,
        "log_manipulability_start": float(trace["dexterity"][0, 0]),
        "log_manipulability_end": float(trace["dexterity"][-1, 0]),
        "scaled_sigma_min_p05": float(np.percentile(trace["dexterity"][:, 1], 5)),
        "scaled_sigma_min_end": float(trace["dexterity"][-1, 1]),
        "scaled_volume_end": float(trace["dexterity"][-1, 2]),
        "arm_travel_rad": float(np.abs(np.diff(trace["command_q"][:, q_indices], axis=0)).sum()),
        "duration_s": float(trace["time"][-1]),
    }


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def save_manifest(output: Path, config: OperationalSpaceConfig, scenarios: tuple) -> None:
    """Capture dirty sources too, so git HEAD alone cannot misidentify a run."""
    root = find_repo_root(__file__)
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "uv.lock"], cwd=root, text=True
    ).splitlines()
    changed = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "src"], cwd=root, text=True
    ).splitlines()
    paths = sorted(set(tracked + changed))
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in paths
        if (root / name).is_file()
    }
    (output / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    )
    for name in changed:
        path = output / "untracked_sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root / name).read_bytes())
    write_json(
        output / "manifest.json",
        {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "source_sha256": hashes,
            "osc": asdict(config),
            "gates": asdict(TUNING_GATES),
            "simulation": asdict(DEFAULT_MUJOCO_SIMULATION_CONFIG),
            "scenarios": [asdict(scenario) for scenario in scenarios],
            "versions": {
                name: version(name) for name in ("pin-pink", "pin", "mujoco", "numpy", "quadprog")
            },
            "jacobian_length_scale_m": JACOBIAN_LENGTH_SCALE_M,
            "notes": (
                "200 Hz physics; controller at first physics tick at/after each 30 Hz deadline; "
                "actual integration dt; 2 s initial actuator settle; commanded-state integration; "
                "no network or hardware. No self-contact physics; Pinocchio checks all configured "
                "pairs. Root uses ideal planar velocity actuators."
            ),
        },
    )


def run_experiment(
    output: Path,
    config: OperationalSpaceConfig,
    scenarios: tuple[OSCTuningScenario, ...],
    repeats: int = 1,
) -> list[dict[str, object]]:
    """Save every trial and the exact experiment inputs in a new directory."""
    output.mkdir(parents=True, exist_ok=False)
    save_manifest(output, config, scenarios)
    rows = []
    for repeat in range(repeats):
        for scenario in scenarios:
            trace, metadata = run_scenario(config, scenario)
            metrics = analyze_trial(trace, metadata, config, scenario)
            name = f"r{repeat + 1}_{scenario.name}"
            np.savez_compressed(output / f"{name}.npz", allow_pickle=False, **trace)
            row = {"trial": name, "scenario": scenario.name, "repeat": repeat + 1, **metrics}
            write_json(output / f"{name}.json", {"metrics": row, "metadata": metadata})
            rows.append(row)
            write_json(output / "results.json", rows)
            print(json.dumps(row, default=_json_default), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--suite",
        choices=("screen", "training", "validation", "confirmation", "robustness"),
        default="screen",
    )
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--output", type=Path, default=Path("logs/osc_tuning"))
    args = parser.parse_args()
    overrides = {key: float(value) for key, value in (item.split("=", 1) for item in args.set)}
    config = replace(BASELINE_OSC_CONFIG, **overrides)
    scenarios = tuning_scenarios(args.suite)
    if args.scenario:
        scenarios = tuple(scenario for scenario in scenarios if scenario.name in args.scenario)
    if not scenarios or args.repeat < 1:
        parser.error("Select at least one scenario and a positive repeat count.")
    run_experiment(args.output / args.name, config, scenarios, args.repeat)


if __name__ == "__main__":
    main()
