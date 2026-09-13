"""Command-line entrypoint for controller tracking."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

from humanoid.config import ROBOT_CONFIG, ROBOT_CONFIGS
from humanoid.logger import get_logger
from humanoid.robots.utils.controller_tracking.models import (
    CARTESIAN_COMPARISON_SEGMENTS,
    DEFAULT_COMMAND_RATE_HZ,
    DEFAULT_COMPARISON_DURATION_SECONDS,
    DEFAULT_CONNECTION_TIMEOUT_SECONDS,
    DEFAULT_CYCLES,
    DEFAULT_FEEDBACK_TIMEOUT_SECONDS,
    DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION,
    DEFAULT_HEIGHT_METERS,
    DEFAULT_HOME_POSITION_TOLERANCE_RAD,
    DEFAULT_HOME_STABLE_SECONDS,
    DEFAULT_HOME_TIMEOUT_SECONDS,
    DEFAULT_JOINT_CYCLES,
    DEFAULT_PERIOD_SECONDS,
    DEFAULT_RAMP_SECONDS,
    DEFAULT_SETTLE_SECONDS,
    DEFAULT_START_DELAY_SECONDS,
    DEFAULT_WIDTH_METERS,
    FIGURE_EIGHT_SEGMENTS,
    JOINT_COMPARISON_SEGMENTS,
    JOINT_HOME_REST_SEGMENTS,
    PLANE_AXES,
    ControllerTrackingSettings,
    Plane,
    TrackingRun,
    TrackingSample,
)
from humanoid.robots.utils.controller_tracking.report import (
    TrackingPlotPaths,
    _cartesian_tracking_table_rows,
    _combined_joint_sample_matrices,
    _format_statistics_table,
    _joint_tracking_table_rows,
    write_controller_timing_csv,
    write_tracking_csv,
    write_tracking_plots,
)
from humanoid.robots.utils.controller_tracking.runtime import run_controller_tracking
from humanoid.robots.utils.controller_tracking.sampling import tracking_statistics
from humanoid.robots.utils.controller_tracking.timing import (
    MINIMUM_PUBLICATION_COUNT,
    controller_publication_statistics,
)
from humanoid.types.controllers import OperationalSpaceConfig
from humanoid.types.robot import RobotName
from humanoid.utils.paths import find_data_root

logger = get_logger(__name__)


def _default_output_directory() -> Path:
    return find_data_root(__file__) / "logs" / "tracking"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure home/rest, Cartesian figure-eight, and start/end comparison tracking."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--plane",
        choices=tuple(PLANE_AXES),
        default="xz",
        help="Projection plane for Cartesian path plots",
    )
    parser.add_argument(
        "--robot",
        type=RobotName,
        choices=list(RobotName),
        default=ROBOT_CONFIG.name,
        help="Robot model; defaults to HUMANOID_ROBOT or the project default",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=DEFAULT_WIDTH_METERS,
        help="Figure-eight width in meters",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=DEFAULT_HEIGHT_METERS,
        help="Figure-eight height in meters",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=DEFAULT_PERIOD_SECONDS,
        help="Duration of one figure-eight cycle",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=DEFAULT_CYCLES,
        help="Figure-eight trajectory repetitions",
    )
    parser.add_argument(
        "--comparison-duration",
        type=float,
        default=DEFAULT_COMPARISON_DURATION_SECONDS,
        help="Duration of the Cartesian start-to-end comparison motion",
    )
    parser.add_argument(
        "--joint-cycles",
        type=int,
        default=DEFAULT_JOINT_CYCLES,
        help="Home-to-rest-to-home baseline round trips; use 0 to omit from the report",
    )
    parser.add_argument(
        "--ramp",
        type=float,
        default=DEFAULT_RAMP_SECONDS,
        help="Duration used to blend figure-eight and gripper motion in and out",
    )
    parser.add_argument("--rate", type=float, default=DEFAULT_COMMAND_RATE_HZ)
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument("--start-delay", type=float, default=DEFAULT_START_DELAY_SECONDS)
    parser.add_argument(
        "--connection-timeout", type=float, default=DEFAULT_CONNECTION_TIMEOUT_SECONDS
    )
    parser.add_argument("--feedback-timeout", type=float, default=DEFAULT_FEEDBACK_TIMEOUT_SECONDS)
    parser.add_argument(
        "--home-tolerance",
        type=float,
        default=DEFAULT_HOME_POSITION_TOLERANCE_RAD,
        help="Maximum arm-joint error in radians before each test",
    )
    parser.add_argument(
        "--home-stable",
        type=float,
        default=DEFAULT_HOME_STABLE_SECONDS,
        help="Time the measured arm must remain within the HOME tolerance",
    )
    parser.add_argument(
        "--home-timeout",
        type=float,
        default=DEFAULT_HOME_TIMEOUT_SECONDS,
        help="Maximum time to wait for measured HOME convergence",
    )
    parser.add_argument(
        "--hold-gripper",
        action="store_true",
        help="Hold the measured gripper opening instead of cycling it",
    )
    parser.add_argument(
        "--gripper-min",
        type=float,
        help="Minimum gripper joint position in radians; requires --gripper-max",
    )
    parser.add_argument(
        "--gripper-max",
        type=float,
        help="Maximum gripper joint position in radians; requires --gripper-min",
    )
    parser.add_argument(
        "--gripper-period",
        type=float,
        help="Gripper sinusoid period in seconds; defaults to the motion period",
    )
    parser.add_argument(
        "--gripper-limit-margin",
        type=float,
        default=DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION,
        help="Fraction of the model range kept clear at each gripper limit",
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_directory())
    return parser


def _settings_from_args(args: argparse.Namespace) -> ControllerTrackingSettings:
    return ControllerTrackingSettings(
        plane=cast(Plane, args.plane),
        width_m=args.width,
        height_m=args.height,
        period_s=args.period,
        cycles=args.cycles,
        comparison_duration_s=args.comparison_duration,
        joint_cycles=args.joint_cycles,
        ramp_s=args.ramp,
        rate_hz=args.rate,
        settle_s=args.settle,
        start_delay_s=args.start_delay,
        connection_timeout_s=args.connection_timeout,
        feedback_timeout_s=args.feedback_timeout,
        home_position_tolerance_rad=args.home_tolerance,
        home_stable_s=args.home_stable,
        home_timeout_s=args.home_timeout,
        move_gripper=not args.hold_gripper,
        gripper_min_rad=args.gripper_min,
        gripper_max_rad=args.gripper_max,
        gripper_period_s=args.gripper_period,
        gripper_limit_margin_fraction=args.gripper_limit_margin,
    )


def _log_run_results(
    run: TrackingRun,
    csv_path: Path,
    timing_csv_path: Path,
    plot_paths: TrackingPlotPaths,
    target_rate_hz: float,
) -> None:
    joint_samples = [sample for sample in run.samples if sample.segment in JOINT_HOME_REST_SEGMENTS]
    if joint_samples:
        joint_stats = tracking_statistics(joint_samples)
        _, _, joint_names = _combined_joint_sample_matrices(joint_samples)
        logger.info(
            "Joint-space home/rest tracking error:\n%s",
            _format_statistics_table(_joint_tracking_table_rows(joint_stats, joint_names)),
        )

    figure_samples = [sample for sample in run.samples if sample.segment == "figure_eight"]
    if figure_samples:
        figure_stats = tracking_statistics(figure_samples)
        _, _, figure_joint_names = _combined_joint_sample_matrices(figure_samples)
        logger.info(
            "Cartesian figure-eight joint tracking error:\n%s",
            _format_statistics_table(_joint_tracking_table_rows(figure_stats, figure_joint_names)),
        )
        logger.info(
            "Cartesian figure-eight tool tracking error:\n%s",
            _format_statistics_table(_cartesian_tracking_table_rows(figure_stats)),
        )
    else:
        logger.warning("Cartesian figure-eight phase was not reached")

    direct_samples = [sample for sample in run.samples if sample.segment == "joint_comparison"]
    if direct_samples:
        direct_stats = tracking_statistics(direct_samples)
        _, _, direct_joint_names = _combined_joint_sample_matrices(direct_samples)
        logger.info(
            "Homing-controller comparison joint tracking error:\n%s",
            _format_statistics_table(_joint_tracking_table_rows(direct_stats, direct_joint_names)),
        )
        logger.info(
            "Homing-controller comparison Cartesian error:\n%s",
            _format_statistics_table(_cartesian_tracking_table_rows(direct_stats)),
        )
    else:
        logger.warning("Homing-controller comparison phase was not reached")

    cartesian_samples = [
        sample for sample in run.samples if sample.segment == "cartesian_comparison"
    ]
    if cartesian_samples:
        cartesian_stats = tracking_statistics(cartesian_samples)
        _, _, cartesian_joint_names = _combined_joint_sample_matrices(cartesian_samples)
        logger.info(
            "Cartesian OSC/IK comparison joint tracking error:\n%s",
            _format_statistics_table(
                _joint_tracking_table_rows(cartesian_stats, cartesian_joint_names)
            ),
        )
        logger.info(
            "Cartesian OSC/IK comparison tracking error:\n%s",
            _format_statistics_table(_cartesian_tracking_table_rows(cartesian_stats)),
        )
    else:
        logger.warning("Cartesian OSC/IK comparison phase was not reached")

    _log_command_to_state_timing(run.samples)
    for label, segments in (
        ("Cartesian figure eight", FIGURE_EIGHT_SEGMENTS),
        ("homing-controller comparison", JOINT_COMPARISON_SEGMENTS),
        ("Cartesian OSC/IK comparison", CARTESIAN_COMPARISON_SEGMENTS),
    ):
        phase_samples = [sample for sample in run.samples if sample.segment in segments]
        if phase_samples:
            final_error_mm = np.linalg.norm(phase_samples[-1].position_error_m) * 1_000.0
            logger.info("Final tool position error after %s: %.2f mm", label, final_error_mm)
    for label, segment in (
        ("Cartesian figure eight", "figure_eight"),
        ("Cartesian OSC/IK comparison", "cartesian_comparison"),
    ):
        phase_timings = [
            timing for timing in run.controller_command_timings if timing.segment == segment
        ]
        if len(phase_timings) < MINIMUM_PUBLICATION_COUNT:
            logger.warning("No complete OSC publication-rate measurement for %s", label)
            continue
        rate_stats = controller_publication_statistics(phase_timings, target_rate_hz)
        logger.info(
            "%s OSC publication rate: %.2f Hz over %d commands; period "
            "median %.2f ms, p95 %.2f ms, max %.2f ms, jitter %.2f ms; "
            "%d delayed interval(s)",
            label,
            rate_stats.mean_rate_hz,
            rate_stats.command_count,
            rate_stats.median_period_s * 1_000.0,
            rate_stats.p95_period_s * 1_000.0,
            rate_stats.maximum_period_s * 1_000.0,
            rate_stats.period_jitter_s * 1_000.0,
            rate_stats.delayed_interval_count,
        )
    logger.info("Raw samples: %s", csv_path)
    logger.info("Raw OSC publication timing: %s", timing_csv_path)
    logger.info("Joint-space HOME/REST plot: %s", plot_paths.home_rest)
    logger.info("Cartesian figure-eight plot: %s", plot_paths.figure_eight)
    logger.info("Command-space comparison plot: %s", plot_paths.comparison)


def _fresh_command_to_state_offsets_s(samples: list[TrackingSample]) -> np.ndarray:
    """Return timestamp offsets only when the observed joint command advances."""
    offsets = []
    previous_command_timestamp_s: float | None = None
    for sample in samples:
        command_timestamp_s = sample.joint_command_timestamp_s
        if (
            previous_command_timestamp_s is None
            or command_timestamp_s > previous_command_timestamp_s
        ):
            offsets.append(sample.state_minus_joint_command_s)
            previous_command_timestamp_s = command_timestamp_s
    return np.asarray(offsets, dtype=float)


def _log_command_to_state_timing(samples: list[TrackingSample]) -> None:
    command_to_state_ms = _fresh_command_to_state_offsets_s(samples) * 1_000.0
    if not command_to_state_ms.size:
        logger.warning("No fresh joint-command timestamps were captured")
        return
    logger.info(
        "State minus fresh joint-command timestamp: median %.1f ms, range [%.1f, %.1f] ms",
        np.median(command_to_state_ms),
        np.min(command_to_state_ms),
        np.max(command_to_state_ms),
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        settings = _settings_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        robot_config = ROBOT_CONFIGS[args.robot]
        run = run_controller_tracking(settings, robot_config)
    except RuntimeError as exc:
        logger.error("Controller tracking measurement failed: %s", exc)
        sys.exit(1)

    if not run.samples:
        logger.info("Controller tracking measurement cancelled before motion began")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"controller_tracking_{robot_config.name}_{timestamp}"
    csv_path = args.output_dir / f"{stem}.csv"
    timing_csv_path = args.output_dir / f"{stem}_controller_timing.csv"
    controller_config = robot_config.operational_space_config or OperationalSpaceConfig()
    target_controller_rate_hz = 1.0 / controller_config.dt
    write_tracking_csv(csv_path, run.samples)
    write_controller_timing_csv(
        timing_csv_path,
        run.controller_command_timings,
        target_controller_rate_hz,
    )
    plot_paths = write_tracking_plots(
        args.output_dir,
        stem,
        run.samples,
        settings,
        robot_config.name.value,
        completed=run.completed,
        failure_reason=run.failure_reason,
    )

    _log_run_results(
        run,
        csv_path,
        timing_csv_path,
        plot_paths,
        target_controller_rate_hz,
    )
    if run.failure_reason is not None:
        logger.error("Report contains a partial run: %s", run.failure_reason)
        sys.exit(1)
    if not run.completed:
        logger.warning("Report contains a partial run because the test was interrupted")
