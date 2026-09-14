"""CSV and SVG report generation for controller tracking."""

import csv
import html
import json
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from humanoid.robots.utils.controller_tracking.models import (
    AXIS_NAMES,
    BOUNDS_EPSILON,
    CARTESIAN_COMPARISON_SEGMENTS,
    DEFAULT_MAXIMUM_SPECTRUM_HZ,
    FIGURE_EIGHT_SEGMENTS,
    JOINT_COMPARISON_SEGMENTS,
    JOINT_HOME_REST_SEGMENTS,
    MAX_PLOT_POINTS,
    MIN_JOINT_PLOT_SPAN_RAD,
    MIN_JOINT_VELOCITY_PLOT_SPAN_RAD_S,
    MIN_PLOT_PADDING,
    PLANE_AXES,
    PLOT_PADDING_FRACTION,
    ControllerTrackingSettings,
    ErrorStatistics,
    Segment,
    StatisticsTableRow,
    TrackingSample,
    TrackingStatistics,
)
from humanoid.robots.utils.controller_tracking.sampling import (
    _arm_joint_sample_matrices,
    _gripper_sample_matrices,
    tracking_statistics,
)
from humanoid.robots.utils.controller_tracking.smoothness import (
    analyze_smoothness,
    smoothness_metrics,
)
from humanoid.types.controller_tracking import (
    ControllerCommandTiming,
    JointSmoothnessStatistics,
    MotionTrace,
    NativeJointSample,
    SmoothnessAnalysis,
)


def write_tracking_csv(path: Path, samples: list[TrackingSample]) -> None:
    """Write raw tracking samples for subsequent analysis."""
    arm_joint_commands, _, _ = _arm_joint_sample_matrices(samples)
    joint_velocity_commands, joint_velocity_measurements, joint_velocity_names = (
        _combined_joint_velocity_matrices(samples)
    )
    arm_joint_names = samples[0].arm_joint_names if samples else ()
    arm_joint_count = arm_joint_commands.shape[1]
    _, _, gripper_errors = _gripper_sample_matrices(samples)
    gripper_joint_names = samples[0].gripper_joint_names if samples else ()
    gripper_count = len(gripper_joint_names)
    joint_names = (*arm_joint_names, *gripper_joint_names)
    if joint_velocity_names != joint_names:
        raise ValueError("Joint velocity values must match the configured joint names")
    if gripper_errors.shape[1] != gripper_count:
        raise ValueError("Gripper tracking values must match the configured joint names")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        header = [
            "segment",
            "elapsed_s",
            "state_timestamp_s",
            "joint_command_timestamp_s",
            "state_minus_joint_command_s",
            "command_x_m",
            "command_y_m",
            "command_z_m",
            "measured_x_m",
            "measured_y_m",
            "measured_z_m",
            "error_x_m",
            "error_y_m",
            "error_z_m",
            "position_error_m",
            "orientation_error_rad",
            "feedforward_vx_m_s",
            "feedforward_vy_m_s",
            "feedforward_vz_m_s",
            "feedforward_wx_rad_s",
            "feedforward_wy_rad_s",
            "feedforward_wz_rad_s",
        ]
        for joint_name in arm_joint_names:
            header.extend(
                [
                    f"{joint_name}_controller_command_rad",
                    f"{joint_name}_measured_rad",
                    f"{joint_name}_error_rad",
                    f"{joint_name}_controller_command_rad_s",
                    f"{joint_name}_measured_rad_s",
                ]
            )
        for joint_name in gripper_joint_names:
            header.extend(
                [
                    f"{joint_name}_controller_command_rad",
                    f"{joint_name}_measured_rad",
                    f"{joint_name}_error_rad",
                    f"{joint_name}_controller_command_rad_s",
                    f"{joint_name}_measured_rad_s",
                ]
            )
        writer.writerow(header)
        for sample_index, sample in enumerate(samples):
            row: list[object] = [
                sample.segment,
                sample.elapsed_s,
                sample.state_timestamp_s,
                sample.joint_command_timestamp_s,
                sample.state_minus_joint_command_s,
                *sample.commanded_position_m,
                *sample.measured_position_m,
                *sample.position_error_m,
                np.linalg.norm(sample.position_error_m),
                sample.orientation_error_rad,
                *(
                    sample.commanded_linear_velocity_m_s
                    if sample.commanded_linear_velocity_m_s is not None
                    else ("", "", "")
                ),
                *(
                    sample.commanded_angular_velocity_rad_s
                    if sample.commanded_angular_velocity_rad_s is not None
                    else ("", "", "")
                ),
            ]
            for index in range(arm_joint_count):
                row.extend(
                    [
                        sample.commanded_arm_joint_positions_rad[index],
                        sample.measured_arm_joint_positions_rad[index],
                        sample.arm_joint_position_errors_rad[index],
                        joint_velocity_commands[sample_index, index],
                        joint_velocity_measurements[sample_index, index],
                    ]
                )
            if gripper_count:
                assert sample.commanded_gripper_positions_rad is not None
                assert sample.measured_gripper_positions_rad is not None
                assert sample.gripper_position_errors_rad is not None
                for index in range(gripper_count):
                    row.extend(
                        [
                            sample.commanded_gripper_positions_rad[index],
                            sample.measured_gripper_positions_rad[index],
                            sample.gripper_position_errors_rad[index],
                            joint_velocity_commands[sample_index, arm_joint_count + index],
                            joint_velocity_measurements[sample_index, arm_joint_count + index],
                        ]
                    )
            writer.writerow(row)


def write_controller_timing_csv(
    path: Path,
    timings: list[ControllerCommandTiming],
    target_rate_hz: float,
) -> None:
    """Write every captured controller and routed robot publication interval."""
    if not math.isfinite(target_rate_hz) or target_rate_hz <= 0.0:
        raise ValueError("Target publication rate must be positive and finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    first_timestamp_by_window: dict[tuple[str, Segment, int], float] = {}
    previous_timestamp_by_window: dict[tuple[str, Segment, int], float] = {}
    delayed_threshold_s = 1.5 / target_rate_hz
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "stream",
                "segment",
                "window_index",
                "received_timestamp_s",
                "source_timestamp_s",
                "segment_elapsed_s",
                "period_s",
                "instantaneous_rate_hz",
                "delayed_interval",
            ]
        )
        for timing in sorted(timings, key=lambda sample: (sample.stream, sample.timestamp_s)):
            stream_window = (timing.stream, timing.segment, timing.window_index)
            first_timestamp = first_timestamp_by_window.setdefault(
                stream_window,
                timing.timestamp_s,
            )
            previous_timestamp = previous_timestamp_by_window.get(stream_window)
            period_s = (
                timing.timestamp_s - previous_timestamp if previous_timestamp is not None else None
            )
            if period_s is not None and period_s <= 0.0:
                raise ValueError(
                    "Controller publication timestamps must be unique and increasing per segment"
                )
            writer.writerow(
                [
                    timing.stream,
                    timing.segment,
                    timing.window_index,
                    timing.timestamp_s,
                    "" if timing.source_timestamp_s is None else timing.source_timestamp_s,
                    timing.timestamp_s - first_timestamp,
                    "" if period_s is None else period_s,
                    "" if period_s is None else 1.0 / period_s,
                    "" if period_s is None else period_s > delayed_threshold_s,
                ]
            )
            previous_timestamp_by_window[stream_window] = timing.timestamp_s


def write_native_joint_telemetry_csv(path: Path, samples: list[NativeJointSample]) -> None:
    """Write every native-rate controller command, routed command, and robot state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "stream",
                "segment",
                "window_index",
                "received_timestamp_s",
                "source_timestamp_s",
                "joint_name",
                "position_rad",
                "velocity_rad_s",
                "tool_x_m",
                "tool_y_m",
                "tool_z_m",
            ]
        )
        for sample in samples:
            for joint_index, joint_name in enumerate(sample.joint_names):
                velocity = (
                    ""
                    if sample.joint_velocities_rad_s is None
                    else sample.joint_velocities_rad_s[joint_index]
                )
                writer.writerow(
                    [
                        sample.stream,
                        sample.segment,
                        sample.window_index,
                        sample.received_timestamp_s,
                        sample.source_timestamp_s,
                        joint_name,
                        sample.joint_positions_rad[joint_index],
                        velocity,
                        *sample.tool_position_m,
                    ]
                )


def write_run_metrics_json(
    path: Path,
    samples: list[TrackingSample],
    analyses: dict[str, SmoothnessAnalysis],
) -> None:
    """Write machine-readable tracking and smoothness results for A/B comparison."""
    tracking = {}
    phase_segments: tuple[tuple[str, frozenset[Segment]], ...] = (
        ("home_rest", frozenset({"joint_home", "joint_rest"})),
        ("figure_eight", frozenset({"figure_eight"})),
        ("joint_comparison", frozenset({"joint_comparison"})),
        ("cartesian_comparison", frozenset({"cartesian_comparison"})),
    )
    for phase_name, segments in phase_segments:
        phase_samples = [sample for sample in samples if sample.segment in segments]
        if not phase_samples:
            continue
        stats = tracking_statistics(phase_samples)
        _, _, joint_names = _combined_joint_sample_matrices(phase_samples)
        joint_stats = (*stats.arm_joint_position_rad, *stats.gripper_position_rad)
        tracking[phase_name] = {
            "joints": {
                name: _error_statistics_payload(error_stats)
                for name, error_stats in zip(joint_names, joint_stats, strict=True)
            },
            "tool_translation_m": _error_statistics_payload(stats.position_m),
            "tool_orientation_rad": _error_statistics_payload(stats.orientation_rad),
        }
    payload = {
        "tracking": tracking,
        "smoothness": {name: smoothness_metrics(analysis) for name, analysis in analyses.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _error_statistics_payload(stats: ErrorStatistics) -> dict[str, float]:
    return {
        "rms": stats.rms,
        "mean": stats.mean,
        "p95": stats.p95,
        "maximum": stats.maximum,
    }


def build_smoothness_analyses(
    samples: list[NativeJointSample],
    settings: ControllerTrackingSettings,
    acceleration_limits_rad_s2: dict[str, float] | None,
) -> dict[str, SmoothnessAnalysis]:
    """Analyze each reported phase using motion and settle windows separately."""
    configurations: tuple[tuple[str, tuple[Segment, ...], tuple[Segment, ...], bool], ...] = (
        ("home_rest", ("joint_home", "joint_rest"), ("joint_home_settle",), False),
        ("figure_eight", ("figure_eight",), ("figure_eight_settle",), True),
        (
            "joint_comparison",
            ("joint_comparison",),
            ("joint_comparison_settle",),
            False,
        ),
        (
            "cartesian_comparison",
            ("cartesian_comparison",),
            ("cartesian_comparison_settle",),
            True,
        ),
    )
    analyses = {}
    for name, motion_segments, settle_segments, uses_osc_limits in configurations:
        analysis = analyze_smoothness(
            samples,
            motion_segments=motion_segments,
            settle_segments=settle_segments,
            settings=settings,
            acceleration_limits_rad_s2=(acceleration_limits_rad_s2 if uses_osc_limits else None),
        )
        if analysis is not None:
            analyses[name] = analysis
    return analyses


@dataclass(frozen=True, kw_only=True)
class TrackingPlotPaths:
    """Paths for the three independently rendered tracking reports."""

    home_rest: Path
    figure_eight: Path
    comparison: Path

    def __iter__(self) -> Iterator[Path]:
        return iter((self.home_rest, self.figure_eight, self.comparison))


def write_tracking_plots(  # noqa: PLR0913 - output and report inputs are independent
    output_directory: Path,
    stem: str,
    samples: list[TrackingSample],
    settings: ControllerTrackingSettings,
    robot_name: str,
    *,
    native_joint_samples: list[NativeJointSample] | None = None,
    acceleration_limits_rad_s2: dict[str, float] | None = None,
    completed: bool = True,
    failure_reason: str | None = None,
) -> TrackingPlotPaths:
    """Render one standalone SVG report for each tracking experiment."""
    if not samples:
        raise ValueError("Cannot plot an empty tracking run")

    paths = TrackingPlotPaths(
        home_rest=output_directory / f"{stem}_home_rest.svg",
        figure_eight=output_directory / f"{stem}_figure_eight.svg",
        comparison=output_directory / f"{stem}_comparison.svg",
    )
    write_home_rest_plot(
        paths.home_rest,
        samples,
        settings,
        robot_name,
        native_joint_samples=native_joint_samples,
        acceleration_limits_rad_s2=acceleration_limits_rad_s2,
        completed=completed,
        failure_reason=failure_reason,
    )
    write_figure_eight_plot(
        paths.figure_eight,
        samples,
        settings,
        robot_name,
        native_joint_samples=native_joint_samples,
        acceleration_limits_rad_s2=acceleration_limits_rad_s2,
        completed=completed,
        failure_reason=failure_reason,
    )
    write_comparison_plot(
        paths.comparison,
        samples,
        settings,
        robot_name,
        native_joint_samples=native_joint_samples,
        acceleration_limits_rad_s2=acceleration_limits_rad_s2,
        completed=completed,
        failure_reason=failure_reason,
    )
    return paths


def write_home_rest_plot(  # noqa: PLR0913 - output and report inputs are independent
    path: Path,
    samples: list[TrackingSample],
    settings: ControllerTrackingSettings,
    robot_name: str,
    *,
    native_joint_samples: list[NativeJointSample] | None = None,
    acceleration_limits_rad_s2: dict[str, float] | None = None,
    completed: bool = True,
    failure_reason: str | None = None,
) -> None:
    """Render joint-space HOME/REST tracking."""
    joint_samples = [sample for sample in samples if sample.segment in JOINT_HOME_REST_SEGMENTS]
    elements: list[str] = []
    phase_heading_y = _append_report_heading(
        elements,
        title="Joint-space home/rest tracking",
        completed=completed,
    )
    phase_content_top = phase_heading_y + 50.0
    if joint_samples:
        smoothness = analyze_smoothness(
            native_joint_samples or [],
            motion_segments=("joint_home", "joint_rest"),
            settle_segments=("joint_home_settle",),
            settings=settings,
            acceleration_limits_rad_s2=None,
        )
        content_bottom = _append_joint_tracking_block(
            elements,
            summary_samples=joint_samples,
            plot_samples=joint_samples,
            table_top=phase_content_top,
            separator_note="Dashed lines mark HOME/REST transitions.",
            smoothness=smoothness,
            acceleration_limits_rad_s2=None,
        )
    else:
        content_bottom = _append_empty_tracking_phase(
            elements,
            top=phase_content_top,
            message="Joint-space tracking was disabled or was not reached.",
        )
    _write_svg_document(
        path,
        elements,
        content_bottom,
        settings,
        robot_name,
        completed=completed,
        failure_reason=failure_reason,
    )


def write_figure_eight_plot(  # noqa: PLR0913 - output and report inputs are independent
    path: Path,
    samples: list[TrackingSample],
    settings: ControllerTrackingSettings,
    robot_name: str,
    *,
    native_joint_samples: list[NativeJointSample] | None = None,
    acceleration_limits_rad_s2: dict[str, float] | None = None,
    completed: bool = True,
    failure_reason: str | None = None,
) -> None:
    """Render joint and Cartesian tracking for the figure-eight experiment."""
    figure_samples = [sample for sample in samples if sample.segment in FIGURE_EIGHT_SEGMENTS]
    figure_trajectory_samples = [
        sample for sample in figure_samples if sample.segment == "figure_eight"
    ]
    elements: list[str] = []
    phase_heading_y = _append_report_heading(
        elements,
        title="Cartesian-space figure eight (OSC/IK)",
        completed=completed,
    )
    phase_content_top = phase_heading_y + 50.0
    if figure_samples:
        figure_summary_samples = figure_trajectory_samples or figure_samples
        smoothness = analyze_smoothness(
            native_joint_samples or [],
            motion_segments=("figure_eight",),
            settle_segments=("figure_eight_settle",),
            settings=settings,
            acceleration_limits_rad_s2=acceleration_limits_rad_s2,
        )
        content_bottom = _append_joint_tracking_block(
            elements,
            summary_samples=figure_summary_samples,
            plot_samples=figure_samples,
            table_top=phase_content_top,
            separator_note="Dashed lines mark the transition to the settle interval.",
            smoothness=smoothness,
            acceleration_limits_rad_s2=acceleration_limits_rad_s2,
        )
        content_bottom = _append_cartesian_tracking_block(
            elements,
            summary_samples=figure_summary_samples,
            plot_samples=figure_samples,
            settings=settings,
            table_top=content_bottom + 65.0,
            trajectory_segment="figure_eight",
            settle_segment="figure_eight_settle",
        )
    else:
        content_bottom = _append_empty_tracking_phase(
            elements,
            top=phase_content_top,
            message="The Cartesian figure-eight phase was not reached.",
        )
    _write_svg_document(
        path,
        elements,
        content_bottom,
        settings,
        robot_name,
        completed=completed,
        failure_reason=failure_reason,
    )


def write_comparison_plot(  # noqa: PLR0913 - report inputs are independent
    path: Path,
    samples: list[TrackingSample],
    settings: ControllerTrackingSettings,
    robot_name: str,
    *,
    native_joint_samples: list[NativeJointSample] | None = None,
    acceleration_limits_rad_s2: dict[str, float] | None = None,
    completed: bool = True,
    failure_reason: str | None = None,
) -> None:
    """Render the joint-controller versus OSC/IK endpoint comparison."""
    direct_samples = [sample for sample in samples if sample.segment in JOINT_COMPARISON_SEGMENTS]
    direct_trajectory_samples = [
        sample for sample in direct_samples if sample.segment == "joint_comparison"
    ]
    cartesian_samples = [
        sample for sample in samples if sample.segment in CARTESIAN_COMPARISON_SEGMENTS
    ]
    cartesian_trajectory_samples = [
        sample for sample in cartesian_samples if sample.segment == "cartesian_comparison"
    ]
    elements: list[str] = []
    phase_heading_y = _append_report_heading(
        elements,
        title="Same-endpoint command-space comparison",
        completed=completed,
    )
    comparison_note_y = phase_heading_y + 27.0
    direct_heading_y = phase_heading_y + 72.0
    elements.extend(
        [
            (
                f'<text class="section-note" x="75" y="{comparison_note_y:.2f}">'
                "Both paths move between the supplied corresponding joint/task poses.</text>"
            ),
            (
                f'<text class="subsection-title" x="75" y="{direct_heading_y:.2f}">'
                "A · Joint-space trajectory (homing controller)</text>"
            ),
        ]
    )
    phase_content_top = direct_heading_y + 42.0
    if direct_samples:
        direct_summary_samples = direct_trajectory_samples or direct_samples
        direct_smoothness = analyze_smoothness(
            native_joint_samples or [],
            motion_segments=("joint_comparison",),
            settle_segments=("joint_comparison_settle",),
            settings=settings,
            acceleration_limits_rad_s2=None,
        )
        content_bottom = _append_joint_tracking_block(
            elements,
            summary_samples=direct_summary_samples,
            plot_samples=direct_samples,
            table_top=phase_content_top,
            separator_note="Dashed lines mark the transition to the settle interval.",
            smoothness=direct_smoothness,
            acceleration_limits_rad_s2=None,
        )
        content_bottom = _append_cartesian_tracking_block(
            elements,
            summary_samples=direct_summary_samples,
            plot_samples=direct_samples,
            settings=settings,
            table_top=content_bottom + 65.0,
            trajectory_segment="joint_comparison",
            settle_segment="joint_comparison_settle",
        )
    else:
        content_bottom = _append_empty_tracking_phase(
            elements,
            top=phase_content_top,
            message="The homing-controller comparison phase was not reached.",
        )

    cartesian_heading_y = content_bottom + 75.0
    elements.extend(
        [
            (
                f'<line class="subsection-divider" x1="75" '
                f'y1="{cartesian_heading_y - 38.0:.2f}" x2="1205" '
                f'y2="{cartesian_heading_y - 38.0:.2f}"/>'
            ),
            (
                f'<text class="subsection-title" x="75" y="{cartesian_heading_y:.2f}">'
                "B · Cartesian trajectory (OSC/IK)</text>"
            ),
        ]
    )
    phase_content_top = cartesian_heading_y + 42.0
    if cartesian_samples:
        cartesian_summary_samples = cartesian_trajectory_samples or cartesian_samples
        cartesian_smoothness = analyze_smoothness(
            native_joint_samples or [],
            motion_segments=("cartesian_comparison",),
            settle_segments=("cartesian_comparison_settle",),
            settings=settings,
            acceleration_limits_rad_s2=acceleration_limits_rad_s2,
        )
        content_bottom = _append_joint_tracking_block(
            elements,
            summary_samples=cartesian_summary_samples,
            plot_samples=cartesian_samples,
            table_top=phase_content_top,
            separator_note="Dashed lines mark the transition to the settle interval.",
            smoothness=cartesian_smoothness,
            acceleration_limits_rad_s2=acceleration_limits_rad_s2,
        )
        content_bottom = _append_cartesian_tracking_block(
            elements,
            summary_samples=cartesian_summary_samples,
            plot_samples=cartesian_samples,
            settings=settings,
            table_top=content_bottom + 65.0,
            trajectory_segment="cartesian_comparison",
            settle_segment="cartesian_comparison_settle",
        )
    else:
        content_bottom = _append_empty_tracking_phase(
            elements,
            top=phase_content_top,
            message="The Cartesian OSC/IK comparison phase was not reached.",
        )

    _write_svg_document(
        path,
        elements,
        content_bottom,
        settings,
        robot_name,
        completed=completed,
        failure_reason=failure_reason,
    )


def _append_report_heading(
    elements: list[str],
    *,
    title: str,
    completed: bool,
) -> float:
    heading_y = 145.0 if not completed else 125.0
    elements.append(
        f'<text class="phase-title" x="75" y="{heading_y:.2f}">{html.escape(title)}</text>'
    )
    return heading_y


def _write_svg_document(  # noqa: PLR0913 - document metadata is independent
    path: Path,
    elements: list[str],
    content_bottom: float,
    settings: ControllerTrackingSettings,
    robot_name: str,
    *,
    completed: bool,
    failure_reason: str | None,
) -> None:
    """Wrap report content in shared styling and write it to disk."""

    canvas_height = math.ceil(content_bottom + 55.0)
    partial_run_status = []
    if not completed:
        detail = failure_reason or "motion stopped before all phases completed"
        partial_run_status.append(
            f'<text class="section-note" x="40" y="96">Partial run — {html.escape(detail)}</text>'
        )
    document = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{canvas_height}" '
        f'viewBox="0 0 1280 {canvas_height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172033}",
        ".title{font-size:27px;font-weight:700}.subtitle{font-size:14px;fill:#586174}",
        ".phase-title{font-size:22px;font-weight:750}.subsection-title{font-size:18px;font-weight:700}",
        ".section-note{font-size:12px;fill:#677286}.panel-title{font-size:15px;font-weight:650}",
        ".axis{stroke:#9ca7b8;stroke-width:1}.grid{stroke:#dfe4eb;stroke-width:1}",
        ".tick{font-size:11px;fill:#677286}.label{font-size:12px;fill:#3e485a}",
        ".table-bg,.empty-bg{fill:#fff;stroke:#d8dee8}.table-head{font-size:11px;font-weight:700;fill:#677286}",
        ".table-label{font-size:12px;font-weight:650}.table-value{font-size:12px;fill:#3e485a}",
        ".table-line{stroke:#e6eaf0;stroke-width:1}.phase-divider{stroke:#cfd6e1;stroke-width:1}",
        ".subsection-divider{stroke:#dce2ea;stroke-width:1}",
        ".command,.joint-command{fill:none;stroke:#2667d8;stroke-width:2.3}",
        ".measured,.joint-measured{fill:none;stroke:#e07a2d;stroke-width:2.1}",
        ".joint-measured{stroke-width:1.8;stroke-dasharray:5 3}",
        ".error{fill:none;stroke:#a33bc1;stroke-width:2.1}",
        ".velocity-x{fill:none;stroke:#2667d8;stroke-width:2.0}",
        ".velocity-y{fill:none;stroke:#31a36b;stroke-width:2.0}",
        ".velocity-z{fill:none;stroke:#d6604d;stroke-width:2.0}",
        ".raw-velocity{fill:none;stroke:#a5afbd;stroke-width:1.3;stroke-dasharray:3 3}",
        ".filtered-velocity{fill:none;stroke:#e07a2d;stroke-width:2.0}",
        ".high-frequency{fill:none;stroke:#a33bc1;stroke-width:1.7}",
        ".acceleration-command{fill:none;stroke:#2667d8;stroke-width:1.8}",
        ".acceleration-measured{fill:none;stroke:#e07a2d;stroke-width:1.7}",
        ".acceleration-limit{stroke:#c33;stroke-width:1;stroke-dasharray:5 4}",
        ".spectrum{fill:none;stroke:#5b55b7;stroke-width:1.6}",
        ".separator{stroke:#8b95a6;stroke-width:1;stroke-dasharray:5 5}",
        "</style>",
        f'<rect width="1280" height="{canvas_height}" fill="#f7f9fc"/>',
        '<text class="title" x="40" y="44">Controller tracking measurement</text>',
        (
            f'<text class="subtitle" x="40" y="70">{html.escape(robot_name)} · '
            f"{settings.plane.upper()} plane · {settings.rate_hz:g} Hz · "
            f"{settings.cycles} figure-eight cycles · "
            f"{settings.comparison_duration_s:g} s comparison · "
            f"velocity feedforward {'on' if settings.velocity_feedforward else 'off'} · "
            f"{settings.joint_cycles} home/rest round trips</text>"
        ),
        *partial_run_status,
        *elements,
        "</svg>",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(document), encoding="utf-8")


def _append_joint_tracking_block(  # noqa: PLR0913 - report inputs are independent
    elements: list[str],
    *,
    summary_samples: list[TrackingSample],
    plot_samples: list[TrackingSample],
    table_top: float,
    separator_note: str,
    smoothness: SmoothnessAnalysis | None = None,
    acceleration_limits_rad_s2: dict[str, float] | None = None,
) -> float:
    """Append a joint error table and position/velocity plots for one phase."""
    stats = tracking_statistics(summary_samples)
    _, _, summary_names = _combined_joint_sample_matrices(summary_samples)
    commands, measurements, plot_names = _aligned_joint_position_matrices(plot_samples)
    if summary_names != plot_names:
        raise ValueError("Joint names must remain consistent within a tracking phase")

    rows = _joint_tracking_table_rows(stats, summary_names)
    _append_statistics_table(
        elements,
        left=75.0,
        top=table_top,
        width=1130.0,
        title="Joint tracking error",
        rows=rows,
    )
    table_bottom = table_top + _statistics_table_height(len(rows))
    if not plot_names:
        return _append_empty_tracking_phase(
            elements,
            top=table_bottom + 45.0,
            message="No position-controlled joints were present in this phase.",
        )

    position_bottom = _append_joint_curve_grid(
        elements,
        samples=plot_samples,
        names=plot_names,
        commands=commands,
        measurements=measurements,
        heading_y=table_bottom + 60.0,
        title="Commanded vs measured joint positions",
        note=separator_note,
        y_label="joint position (rad)",
        minimum_span=MIN_JOINT_PLOT_SPAN_RAD,
        include_zero=False,
    )
    velocity_commands, velocity_measurements, velocity_names = _combined_joint_velocity_matrices(
        plot_samples
    )
    if velocity_names != plot_names:
        raise ValueError("Joint position and velocity names must match")
    velocity_bottom = _append_joint_curve_grid(
        elements,
        samples=plot_samples,
        names=velocity_names,
        commands=velocity_commands,
        measurements=velocity_measurements,
        heading_y=position_bottom + 35.0,
        title="Commanded vs measured joint velocities",
        note=(
            "Homing command velocity is inferred from successive position targets when the "
            "controller does not publish velocity."
        ),
        y_label="joint velocity (rad/s)",
        minimum_span=MIN_JOINT_VELOCITY_PLOT_SPAN_RAD_S,
        include_zero=True,
    )
    if smoothness is None:
        return velocity_bottom
    return _append_smoothness_block(
        elements,
        analysis=smoothness,
        heading_y=velocity_bottom + 35.0,
        acceleration_limits_rad_s2=acceleration_limits_rad_s2 or {},
    )


def _append_joint_curve_grid(  # noqa: PLR0913 - plot data and presentation are independent
    elements: list[str],
    *,
    samples: list[TrackingSample],
    names: tuple[str, ...],
    commands: NDArray[np.float64],
    measurements: NDArray[np.float64],
    heading_y: float,
    title: str,
    note: str,
    y_label: str,
    minimum_span: float,
    include_zero: bool,
) -> float:
    """Append a two-column grid of commanded/measured joint curves."""
    expected_shape = (len(samples), len(names))
    if commands.shape != expected_shape or measurements.shape != expected_shape:
        raise ValueError("Joint plot values must match the samples and joint names")
    elements.extend(
        [
            (
                f'<text class="subsection-title" x="75" y="{heading_y:.2f}">'
                f"{html.escape(title)}</text>"
            ),
            (
                f'<text class="section-note" x="75" y="{heading_y + 23.0:.2f}">'
                f"{html.escape(note)}</text>"
            ),
        ]
    )
    _append_legend(elements, 875.0, heading_y)

    times = _relative_times(samples)
    time_bounds = _padded_bounds(times, include_zero=True)
    panel_width = 515.0
    panel_height = 145.0
    row_stride = 240.0
    plots_top = heading_y + 65.0
    for index, joint_name in enumerate(names):
        panel = (
            75.0 if index % 2 == 0 else 690.0,
            plots_top + (index // 2) * row_stride,
            panel_width,
            panel_height,
        )
        value_bounds = _padded_bounds(
            np.concatenate((commands[:, index], measurements[:, index])),
            include_zero=include_zero,
            minimum_span=minimum_span,
        )
        _append_axes(
            elements,
            panel,
            time_bounds,
            value_bounds,
            "elapsed time (s)",
            y_label,
            joint_name,
        )
        elements.extend(
            [
                _polyline(
                    np.column_stack((times, commands[:, index])),
                    panel,
                    time_bounds,
                    value_bounds,
                    "joint-command",
                ),
                _polyline(
                    np.column_stack((times, measurements[:, index])),
                    panel,
                    time_bounds,
                    value_bounds,
                    "joint-measured",
                ),
            ]
        )
        _append_segment_separators(elements, panel, times, time_bounds, samples)

    plot_rows = math.ceil(len(names) / 2)
    return plots_top + (plot_rows - 1) * row_stride + panel_height + 60.0


def _append_smoothness_block(
    elements: list[str],
    *,
    analysis: SmoothnessAnalysis,
    heading_y: float,
    acceleration_limits_rad_s2: dict[str, float],
) -> float:
    """Append motion-localized vibration metrics and native-rate diagnostic plots."""
    elements.extend(
        [
            f'<text class="subsection-title" x="75" y="{heading_y:.2f}">Motion smoothness</text>',
            (
                f'<text class="section-note" x="75" y="{heading_y + 23.0:.2f}">'
                "Metrics use native-rate telemetry and analyze each motion window independently. "
                "HF means motion above the configured shake cutoff.</text>"
            ),
        ]
    )
    table_top = heading_y + 42.0
    table_bottom = _append_smoothness_table(elements, analysis, table_top)
    tool_note = "Tool-tip high-frequency motion unavailable."
    if (
        analysis.tool_high_frequency_rms_m is not None
        and analysis.tool_high_frequency_peak_to_peak_m is not None
    ):
        tool_note = (
            "Tool-tip high-frequency motion during movement: "
            f"{analysis.tool_high_frequency_rms_m * 1_000.0:.2f} mm RMS, "
            f"{analysis.tool_high_frequency_peak_to_peak_m * 1_000.0:.2f} mm peak-to-peak."
        )
    elements.extend(
        [
            (
                f'<text class="section-note" x="75" y="{table_bottom + 25.0:.2f}">'
                "LIMIT % is the fraction of source-controller samples at or above 90% of "
                "the applicable OSC acceleration limit.</text>"
            ),
            (
                f'<text class="section-note" x="75" y="{table_bottom + 45.0:.2f}">'
                f"{html.escape(tool_note)}</text>"
            ),
        ]
    )

    velocity_bottom = _append_native_trace_grid(
        elements,
        names=analysis.joint_names,
        primary_traces=analysis.state_traces,
        primary_values=lambda trace: trace.raw_velocities_rad_s,
        primary_class="raw-velocity",
        secondary_traces=analysis.state_traces,
        secondary_values=lambda trace: trace.velocities_rad_s,
        secondary_class="filtered-velocity",
        heading_y=table_bottom + 90.0,
        title="Raw servo vs filtered position-derived velocity",
        note=(
            "Raw register velocity is gray; the orange trace is a zero-phase offline estimate "
            "from position."
        ),
        y_label="joint velocity (rad/s)",
        include_zero=True,
    )
    residual_bottom = _append_native_trace_grid(
        elements,
        names=analysis.joint_names,
        primary_traces=analysis.state_traces,
        primary_values=lambda trace: trace.high_frequency_positions_rad * 1_000.0,
        primary_class="high-frequency",
        heading_y=velocity_bottom + 25.0,
        title="Motion-localized high-frequency position residual",
        note="Large bursts identify exactly when measured motion departs from its slow trajectory.",
        y_label="high-frequency position (mrad)",
        include_zero=True,
    )
    acceleration_bottom = _append_native_trace_grid(
        elements,
        names=analysis.joint_names,
        primary_traces=analysis.command_traces,
        primary_values=lambda trace: trace.accelerations_rad_s2,
        primary_class="acceleration-command",
        secondary_traces=analysis.state_traces,
        secondary_values=lambda trace: trace.accelerations_rad_s2,
        secondary_class="acceleration-measured",
        heading_y=residual_bottom + 25.0,
        title="Commanded vs filtered measured acceleration",
        note="Red dashed lines show configured OSC acceleration limits where applicable.",
        y_label="joint acceleration (rad/s²)",
        include_zero=True,
        limits=acceleration_limits_rad_s2,
    )
    return _append_spectrum_grid(
        elements,
        analysis=analysis,
        heading_y=acceleration_bottom + 25.0,
    )


def _append_smoothness_table(
    elements: list[str], analysis: SmoothnessAnalysis, top: float
) -> float:
    row_height = 30.0
    table_top = top + 16.0
    table_height = 50.0 + row_height * len(analysis.statistics)
    numeric_x = (430.0, 565.0, 700.0, 835.0, 960.0, 1080.0, 1190.0)
    headings = ("HF RMS", "HF PK-PK", "ACC P95", "JERK RMS", "LIMIT %", "HOLD PK-PK", "PEAK HZ")
    elements.extend(
        [
            f'<rect class="table-bg" x="75" y="{table_top:.2f}" width="1130" '
            f'height="{table_height:.2f}" rx="8"/>',
            f'<text class="table-head" x="95" y="{top + 42.0:.2f}">JOINT</text>',
        ]
    )
    for x, heading in zip(numeric_x, headings, strict=True):
        elements.append(
            f'<text class="table-head" text-anchor="end" x="{x:.2f}" '
            f'y="{top + 42.0:.2f}">{heading}</text>'
        )
    elements.append(
        f'<line class="table-line" x1="75" y1="{top + 52.0:.2f}" x2="1205" y2="{top + 52.0:.2f}"/>'
    )
    for index, (name, stats) in enumerate(
        zip(analysis.joint_names, analysis.statistics, strict=True)
    ):
        y = top + 76.0 + row_height * index
        elements.append(f'<text class="table-label" x="95" y="{y:.2f}">{html.escape(name)}</text>')
        values = _smoothness_table_values(stats)
        for x, value in zip(numeric_x, values, strict=True):
            elements.append(
                f'<text class="table-value" text-anchor="end" x="{x:.2f}" '
                f'y="{y:.2f}">{html.escape(value)}</text>'
            )
    return table_top + table_height


def _smoothness_table_values(stats: JointSmoothnessStatistics) -> tuple[str, ...]:
    return (
        f"{stats.high_frequency_rms_rad * 1_000.0:.2f} mrad",
        f"{stats.high_frequency_peak_to_peak_rad * 1_000.0:.2f} mrad",
        f"{stats.acceleration_p95_rad_s2:.2f}",
        f"{stats.jerk_rms_rad_s3:.1f}",
        (
            "—"
            if stats.acceleration_limit_fraction is None
            else f"{stats.acceleration_limit_fraction * 100.0:.1f}"
        ),
        (
            "—"
            if stats.settle_peak_to_peak_rad is None
            else f"{stats.settle_peak_to_peak_rad * 1_000.0:.2f} mrad"
        ),
        "—" if stats.dominant_frequency_hz is None else f"{stats.dominant_frequency_hz:.2f}",
    )


def _append_native_trace_grid(  # noqa: PLR0913 - trace presentation is explicit
    elements: list[str],
    *,
    names: tuple[str, ...],
    primary_traces: tuple[MotionTrace, ...],
    primary_values: Callable[[MotionTrace], NDArray[np.float64]],
    primary_class: str,
    heading_y: float,
    title: str,
    note: str,
    y_label: str,
    include_zero: bool,
    secondary_traces: tuple[MotionTrace, ...] = (),
    secondary_values: Callable[[MotionTrace], NDArray[np.float64]] | None = None,
    secondary_class: str = "",
    limits: dict[str, float] | None = None,
) -> float:
    traces = (*primary_traces, *secondary_traces)
    if not traces:
        return heading_y
    time_origin = min(float(trace.times_s[0]) for trace in traces)
    all_times = np.concatenate([trace.times_s - time_origin for trace in traces])
    time_bounds = _padded_bounds(all_times, include_zero=True)
    elements.extend(
        [
            (
                f'<text class="subsection-title" x="75" y="{heading_y:.2f}">'
                f"{html.escape(title)}</text>"
            ),
            (
                f'<text class="section-note" x="75" y="{heading_y + 23.0:.2f}">'
                f"{html.escape(note)}</text>"
            ),
        ]
    )
    plots_top = heading_y + 65.0
    panel_width = 515.0
    panel_height = 145.0
    row_stride = 225.0
    for joint_index, joint_name in enumerate(names):
        panel = (
            75.0 if joint_index % 2 == 0 else 690.0,
            plots_top + (joint_index // 2) * row_stride,
            panel_width,
            panel_height,
        )
        value_arrays = [primary_values(trace)[:, joint_index] for trace in primary_traces]
        if secondary_values is not None:
            value_arrays.extend(
                secondary_values(trace)[:, joint_index] for trace in secondary_traces
            )
        limit = None if limits is None else limits.get(joint_name)
        if limit is not None:
            value_arrays.append(np.array([-limit, limit]))
        value_bounds = _padded_bounds(
            np.concatenate(value_arrays),
            include_zero=include_zero,
        )
        _append_axes(
            elements,
            panel,
            time_bounds,
            value_bounds,
            "elapsed time (s)",
            y_label,
            joint_name,
        )
        elements.extend(
            _polyline(
                np.column_stack(
                    (
                        trace.times_s - time_origin,
                        primary_values(trace)[:, joint_index],
                    )
                ),
                panel,
                time_bounds,
                value_bounds,
                primary_class,
            )
            for trace in primary_traces
        )
        if secondary_values is not None:
            elements.extend(
                _polyline(
                    np.column_stack(
                        (trace.times_s - time_origin, secondary_values(trace)[:, joint_index])
                    ),
                    panel,
                    time_bounds,
                    value_bounds,
                    secondary_class,
                )
                for trace in secondary_traces
            )
        if limit is not None:
            for value in (-limit, limit):
                y = _map_y(value, panel, value_bounds)
                elements.append(
                    f'<line class="acceleration-limit" x1="{panel[0]:.2f}" y1="{y:.2f}" '
                    f'x2="{panel[0] + panel[2]:.2f}" y2="{y:.2f}"/>'
                )
    rows = math.ceil(len(names) / 2)
    return plots_top + (rows - 1) * row_stride + panel_height + 55.0


def _append_spectrum_grid(
    elements: list[str], *, analysis: SmoothnessAnalysis, heading_y: float
) -> float:
    motion_traces = tuple(
        trace
        for trace in analysis.state_traces
        if trace.segment
        not in {
            "joint_home_settle",
            "figure_eight_settle",
            "joint_comparison_settle",
            "cartesian_comparison_settle",
        }
    )
    if not motion_traces:
        return heading_y
    elements.extend(
        [
            (
                f'<text class="subsection-title" x="75" y="{heading_y:.2f}">'
                "Vibration spectrum during motion</text>"
            ),
            (
                f'<text class="section-note" x="75" y="{heading_y + 23.0:.2f}">'
                "Position amplitude spectral density above the shake cutoff; peaks reveal "
                "repeatable shake frequencies.</text>"
            ),
        ]
    )
    plots_top = heading_y + 65.0
    panel_width = 515.0
    panel_height = 145.0
    row_stride = 225.0
    for joint_index, joint_name in enumerate(analysis.joint_names):
        panel = (
            75.0 if joint_index % 2 == 0 else 690.0,
            plots_top + (joint_index // 2) * row_stride,
            panel_width,
            panel_height,
        )
        plotted_values = []
        for trace in motion_traces:
            mask = (trace.frequencies_hz >= analysis.shake_cutoff_hz) & (
                trace.frequencies_hz <= DEFAULT_MAXIMUM_SPECTRUM_HZ
            )
            if np.any(mask):
                plotted_values.append(
                    (
                        trace.frequencies_hz[mask],
                        np.sqrt(trace.position_psd_rad2_hz[mask, joint_index]) * 1_000.0,
                    )
                )
        if not plotted_values:
            continue
        x_bounds = (analysis.shake_cutoff_hz, DEFAULT_MAXIMUM_SPECTRUM_HZ)
        y_bounds = _padded_bounds(
            np.concatenate([values for _, values in plotted_values]),
            include_zero=True,
        )
        _append_axes(
            elements,
            panel,
            x_bounds,
            y_bounds,
            "frequency (Hz)",
            "position ASD (mrad/√Hz)",
            joint_name,
        )
        for frequencies, values in plotted_values:
            elements.append(
                _polyline(
                    np.column_stack((frequencies, values)),
                    panel,
                    x_bounds,
                    y_bounds,
                    "spectrum",
                )
            )
    rows = math.ceil(len(analysis.joint_names) / 2)
    return plots_top + (rows - 1) * row_stride + panel_height + 55.0


def _append_cartesian_tracking_block(  # noqa: PLR0913 - phase and plot inputs are independent
    elements: list[str],
    *,
    summary_samples: list[TrackingSample],
    plot_samples: list[TrackingSample],
    settings: ControllerTrackingSettings,
    table_top: float,
    trajectory_segment: Segment,
    settle_segment: Segment,
) -> float:
    """Append Cartesian error statistics, tool path, and error curves."""
    stats = tracking_statistics(summary_samples)
    rows = _cartesian_tracking_table_rows(stats)
    _append_statistics_table(
        elements,
        left=75.0,
        top=table_top,
        width=1130.0,
        title="Cartesian tracking error",
        rows=rows,
    )
    table_bottom = table_top + _statistics_table_height(len(rows))
    plots_heading = table_bottom + 60.0
    elements.append(
        f'<text class="subsection-title" x="75" y="{plots_heading:.2f}">'
        "Cartesian tracking plots</text>"
    )
    _append_legend(
        elements,
        875.0,
        plots_heading,
        command_class="command",
        measured_class="measured",
    )

    if plot_samples[-1].segment == settle_segment:
        final_error_mm = np.linalg.norm(plot_samples[-1].position_error_m) * 1_000.0
        elements.append(
            f'<text class="section-note" text-anchor="end" x="1205" '
            f'y="{plots_heading + 23.0:.2f}">Final settled tool error: '
            f"{final_error_mm:.2f} mm</text>"
        )

    plots_top = plots_heading + 70.0
    path_panel = (75.0, plots_top, 515.0, 400.0)
    position_panel = (690.0, plots_top, 515.0, 160.0)
    orientation_panel = (690.0, plots_top + 260.0, 515.0, 160.0)
    plane_axes = PLANE_AXES[settings.plane]
    horizontal_axis, vertical_axis = plane_axes

    path_samples = [sample for sample in summary_samples if sample.segment == trajectory_segment]
    path_samples = _decimate(path_samples or summary_samples, MAX_PLOT_POINTS)
    plotted = _decimate(plot_samples, MAX_PLOT_POINTS)
    command_plane = np.array(
        [sample.commanded_position_m[list(plane_axes)] for sample in path_samples]
    )
    measured_plane = np.array(
        [sample.measured_position_m[list(plane_axes)] for sample in path_samples]
    )
    x_bounds, y_bounds = _equal_aspect_bounds(
        np.vstack((command_plane, measured_plane)),
        path_panel[2],
        path_panel[3],
    )
    _append_axes(
        elements,
        path_panel,
        x_bounds,
        y_bounds,
        f"{AXIS_NAMES[horizontal_axis]} position (m)",
        f"{AXIS_NAMES[vertical_axis]} position (m)",
        "Tool path",
    )
    elements.extend(
        [
            _polyline(command_plane, path_panel, x_bounds, y_bounds, "command"),
            _polyline(measured_plane, path_panel, x_bounds, y_bounds, "measured"),
        ]
    )

    times = _relative_times(plotted)
    time_bounds = _padded_bounds(times, include_zero=True)
    position_errors_mm = np.array(
        [np.linalg.norm(sample.position_error_m) * 1_000.0 for sample in plotted]
    )
    orientation_errors_deg = np.rad2deg([sample.orientation_error_rad for sample in plotted])
    _append_error_plot(
        elements,
        panel=position_panel,
        times=times,
        values=position_errors_mm,
        time_bounds=time_bounds,
        y_label="position error (mm)",
        title="Translation error",
    )
    _append_error_plot(
        elements,
        panel=orientation_panel,
        times=times,
        values=orientation_errors_deg,
        time_bounds=time_bounds,
        y_label="orientation error (deg)",
        title="Orientation error",
    )
    for panel in (position_panel, orientation_panel):
        _append_segment_separators(elements, panel, times, time_bounds, plotted)
    content_bottom = plots_top + 500.0
    if any(sample.commanded_linear_velocity_m_s is not None for sample in plotted):
        return _append_feedforward_velocity_plots(
            elements,
            samples=plotted,
            heading_y=content_bottom + 35.0,
        )
    return content_bottom


def _append_feedforward_velocity_plots(
    elements: list[str],
    *,
    samples: list[TrackingSample],
    heading_y: float,
) -> float:
    """Append commanded Cartesian linear and angular feedforward curves."""
    linear_values = [
        sample.commanded_linear_velocity_m_s
        for sample in samples
        if sample.commanded_linear_velocity_m_s is not None
    ]
    angular_values = [
        sample.commanded_angular_velocity_rad_s
        for sample in samples
        if sample.commanded_angular_velocity_rad_s is not None
    ]
    if len(linear_values) != len(samples) or len(angular_values) != len(samples):
        raise ValueError("Cartesian feedforward must be present for every plotted sample")

    linear = np.vstack(linear_values)
    angular = np.vstack(angular_values)
    times = _relative_times(samples)
    time_bounds = _padded_bounds(times, include_zero=True)
    elements.extend(
        [
            (
                f'<text class="subsection-title" x="75" y="{heading_y:.2f}">'
                "Cartesian velocity feedforward</text>"
            ),
            (
                f'<text class="section-note" x="75" y="{heading_y + 23.0:.2f}">'
                "Command-frame reference velocity supplied to OSC/IK.</text>"
            ),
        ]
    )
    _append_axis_legend(elements, left=930.0, baseline=heading_y)

    plots_top = heading_y + 65.0
    panels = (
        ((75.0, plots_top, 515.0, 170.0), linear, "linear velocity (m/s)", "Translation"),
        ((690.0, plots_top, 515.0, 170.0), angular, "angular velocity (rad/s)", "Rotation"),
    )
    for panel, values, y_label, title in panels:
        value_bounds = _padded_bounds(values, include_zero=True)
        _append_axes(
            elements,
            panel,
            time_bounds,
            value_bounds,
            "elapsed time (s)",
            y_label,
            title,
        )
        for axis, css_class in enumerate(("velocity-x", "velocity-y", "velocity-z")):
            elements.append(
                _polyline(
                    np.column_stack((times, values[:, axis])),
                    panel,
                    time_bounds,
                    value_bounds,
                    css_class,
                )
            )
        _append_segment_separators(elements, panel, times, time_bounds, samples)
    return plots_top + 230.0


def _append_axis_legend(elements: list[str], *, left: float, baseline: float) -> None:
    for offset, label, css_class in (
        (0.0, "x", "velocity-x"),
        (70.0, "y", "velocity-y"),
        (140.0, "z", "velocity-z"),
    ):
        elements.extend(
            [
                f'<line x1="{left + offset:.2f}" y1="{baseline - 4.0:.2f}" '
                f'x2="{left + offset + 25.0:.2f}" y2="{baseline - 4.0:.2f}" '
                f'class="{css_class}"/>',
                f'<text class="label" x="{left + offset + 32.0:.2f}" '
                f'y="{baseline:.2f}">{label}</text>',
            ]
        )


def _append_error_plot(  # noqa: PLR0913 - plot geometry and labels are independent
    elements: list[str],
    *,
    panel: tuple[float, float, float, float],
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    time_bounds: tuple[float, float],
    y_label: str,
    title: str,
) -> None:
    value_bounds = _padded_bounds(values, include_zero=True)
    _append_axes(
        elements,
        panel,
        time_bounds,
        value_bounds,
        "elapsed time (s)",
        y_label,
        title,
    )
    elements.append(
        _polyline(
            np.column_stack((times, values)),
            panel,
            time_bounds,
            value_bounds,
            "error",
        )
    )


def _append_empty_tracking_phase(elements: list[str], *, top: float, message: str) -> float:
    height = 90.0
    elements.extend(
        [
            f'<rect class="empty-bg" x="75" y="{top:.2f}" width="1130" '
            f'height="{height:.2f}" rx="8"/>',
            f'<text class="subtitle" text-anchor="middle" x="640" '
            f'y="{top + 50.0:.2f}">{html.escape(message)}</text>',
        ]
    )
    return top + height


def _combined_joint_sample_matrices(
    samples: list[TrackingSample],
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    arm_commands, arm_measurements, _ = _arm_joint_sample_matrices(samples)
    gripper_commands, gripper_measurements, _ = _gripper_sample_matrices(samples)
    commands = np.hstack((arm_commands, gripper_commands))
    measurements = np.hstack((arm_measurements, gripper_measurements))
    arm_names = samples[0].arm_joint_names if samples else ()
    gripper_names = samples[0].gripper_joint_names if samples else ()
    if len(gripper_names) != gripper_commands.shape[1]:
        raise ValueError("Gripper tracking values must match the configured joint names")
    return commands, measurements, (*arm_names, *gripper_names)


def _aligned_joint_position_matrices(
    samples: list[TrackingSample],
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    """Put measurements on the angular branch used by the reported errors."""
    commands, _, names = _combined_joint_sample_matrices(samples)
    _, _, arm_errors = _arm_joint_sample_matrices(samples)
    _, _, gripper_errors = _gripper_sample_matrices(samples)
    errors = np.hstack((arm_errors, gripper_errors))
    if errors.shape != commands.shape:
        raise ValueError("Joint tracking errors must match the joint position commands")
    return commands, commands - errors, names


def _combined_joint_velocity_matrices(
    samples: list[TrackingSample],
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    """Return published-or-inferred command velocities and measured velocities."""
    if not samples:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty.copy(), ()

    arm_names = samples[0].arm_joint_names
    gripper_names = samples[0].gripper_joint_names
    arm_commands, arm_measurements = _joint_velocity_sample_matrices(
        arm_names,
        [sample.commanded_arm_joint_velocities_rad_s for sample in samples],
        [sample.measured_arm_joint_velocities_rad_s for sample in samples],
    )
    gripper_commands, gripper_measurements = _joint_velocity_sample_matrices(
        gripper_names,
        [sample.commanded_gripper_velocities_rad_s for sample in samples],
        [sample.measured_gripper_velocities_rad_s for sample in samples],
    )
    commands = np.hstack((arm_commands, gripper_commands))
    measurements = np.hstack((arm_measurements, gripper_measurements))
    names = (*arm_names, *gripper_names)

    position_commands, _, position_names = _combined_joint_sample_matrices(samples)
    if names != position_names:
        raise ValueError("Joint position and velocity names must match")
    inferred_commands = _infer_command_velocities(samples, position_commands)
    commands = np.where(np.isfinite(commands), commands, inferred_commands)
    return commands, measurements, names


def _joint_velocity_sample_matrices(
    joint_names: tuple[str, ...],
    command_values: list[NDArray[np.float64] | None],
    measurement_values: list[NDArray[np.float64] | None],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate one joint group's velocity samples, retaining missing commands as NaN."""
    sample_count = len(command_values)
    if len(measurement_values) != sample_count:
        raise ValueError("Commanded and measured joint velocity sample counts must match")
    if not joint_names:
        supplied_values = (*command_values, *measurement_values)
        if any(value is not None and value.size for value in supplied_values):
            raise ValueError("Joint velocity values require corresponding joint names")
        empty = np.empty((sample_count, 0), dtype=float)
        return empty, empty.copy()
    expected_shape = (len(joint_names),)
    commands = np.full((sample_count, len(joint_names)), np.nan, dtype=float)
    measurements = np.empty((sample_count, len(joint_names)), dtype=float)
    for index, (command, measurement) in enumerate(
        zip(command_values, measurement_values, strict=True)
    ):
        if command is not None:
            if command.shape != expected_shape or not np.isfinite(command).all():
                raise ValueError("Commanded joint velocities must match the joint names")
            commands[index] = command
        if measurement is None or measurement.shape != expected_shape:
            raise ValueError("Measured joint velocities must match the joint names")
        if not np.isfinite(measurement).all():
            raise ValueError("Measured joint velocities must be finite")
        measurements[index] = measurement
    return commands, measurements


def _infer_command_velocities(
    samples: list[TrackingSample],
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Differentiate position commands independently within each contiguous segment."""
    if positions.shape[0] != len(samples):
        raise ValueError("Command positions must match the tracking samples")
    velocities = np.zeros_like(positions)
    segment_start = 0
    while segment_start < len(samples):
        segment_end = segment_start + 1
        while (
            segment_end < len(samples)
            and samples[segment_end].segment == samples[segment_start].segment
        ):
            segment_end += 1

        unique_indices = [segment_start]
        for index in range(segment_start + 1, segment_end):
            timestamp = samples[index].joint_command_timestamp_s
            previous_timestamp = samples[unique_indices[-1]].joint_command_timestamp_s
            if timestamp < previous_timestamp:
                raise ValueError("Joint command timestamps must be non-decreasing")
            if timestamp > previous_timestamp:
                unique_indices.append(index)

        if len(unique_indices) > 1:
            timestamps = np.array(
                [samples[index].joint_command_timestamp_s for index in unique_indices]
            )
            unwrapped_positions = np.unwrap(positions[unique_indices], axis=0)
            unique_velocities = np.gradient(unwrapped_positions, timestamps, axis=0)
            unique_cursor = 0
            for index in range(segment_start, segment_end):
                while (
                    unique_cursor + 1 < len(unique_indices)
                    and index >= unique_indices[unique_cursor + 1]
                ):
                    unique_cursor += 1
                velocities[index] = unique_velocities[unique_cursor]

        segment_start = segment_end
    return velocities


def _joint_tracking_table_rows(
    stats: TrackingStatistics,
    joint_names: tuple[str, ...],
) -> list[StatisticsTableRow]:
    joint_stats = (*stats.arm_joint_position_rad, *stats.gripper_position_rad)
    return [
        (f"{joint_name} (rad)", error_stats, 1.0)
        for joint_name, error_stats in zip(joint_names, joint_stats, strict=True)
    ]


def _cartesian_tracking_table_rows(stats: TrackingStatistics) -> list[StatisticsTableRow]:
    return [
        ("Tool translation (mm)", stats.position_m, 1_000.0),
        ("Tool orientation (deg)", stats.orientation_rad, 180.0 / np.pi),
    ]


def _statistics_table_height(row_count: int) -> float:
    return 66.0 + 30.0 * row_count


def _format_statistics_table(rows: list[StatisticsTableRow]) -> str:
    label_width = max(len("Signal"), *(len(label) for label, _, _ in rows))
    header = f"{'Signal':<{label_width}}  {'RMS':>8}  {'Mean':>8}  {'P95':>8}  {'Max':>8}"
    separator = "-" * len(header)
    body = []
    for label, stats, scale in rows:
        body.append(
            f"{label:<{label_width}}  {stats.rms * scale:8.2f}  "
            f"{stats.mean * scale:8.2f}  {stats.p95 * scale:8.2f}  "
            f"{stats.maximum * scale:8.2f}"
        )
    return "\n".join((header, separator, *body))


def _append_statistics_table(  # noqa: PLR0913 - SVG table geometry is explicit
    elements: list[str],
    *,
    left: float,
    top: float,
    width: float,
    title: str,
    rows: list[StatisticsTableRow],
) -> None:
    table_top = top + 16.0
    table_height = 50.0 + 30.0 * len(rows)
    label_x = left + 20.0
    numeric_x = tuple(left + width * fraction for fraction in (0.57, 0.70, 0.83, 0.96))
    elements.extend(
        [
            (
                f'<text class="subsection-title" x="{left:.2f}" y="{top:.2f}">'
                f"{html.escape(title)}</text>"
            ),
            f'<rect class="table-bg" x="{left:.2f}" y="{table_top:.2f}" '
            f'width="{width:.2f}" height="{table_height:.2f}" rx="8"/>',
            f'<text class="table-head" x="{label_x:.2f}" y="{top + 42.0:.2f}">SIGNAL</text>',
        ]
    )
    for x, heading in zip(numeric_x, ("RMS", "MEAN", "P95", "MAX"), strict=True):
        elements.append(
            f'<text class="table-head" text-anchor="end" x="{x:.2f}" '
            f'y="{top + 42.0:.2f}">{heading}</text>'
        )
    elements.append(
        f'<line class="table-line" x1="{left:.2f}" y1="{top + 52.0:.2f}" '
        f'x2="{left + width:.2f}" y2="{top + 52.0:.2f}"/>'
    )

    for index, (label, stats, scale) in enumerate(rows):
        y = top + 76.0 + 30.0 * index
        elements.append(
            f'<text class="table-label" x="{label_x:.2f}" y="{y:.2f}">{html.escape(label)}</text>'
        )
        values = (stats.rms, stats.mean, stats.p95, stats.maximum)
        for x, value in zip(numeric_x, values, strict=True):
            elements.append(
                f'<text class="table-value" text-anchor="end" x="{x:.2f}" '
                f'y="{y:.2f}">{value * scale:.2f}</text>'
            )
        if index < len(rows) - 1:
            elements.append(
                f'<line class="table-line" x1="{left:.2f}" y1="{y + 10.0:.2f}" '
                f'x2="{left + width:.2f}" y2="{y + 10.0:.2f}"/>'
            )


def _append_legend(
    elements: list[str],
    left: float,
    baseline: float,
    *,
    command_class: str = "joint-command",
    measured_class: str = "joint-measured",
) -> None:
    elements.extend(
        [
            f'<line x1="{left:.2f}" y1="{baseline - 4.0:.2f}" '
            f'x2="{left + 25.0:.2f}" y2="{baseline - 4.0:.2f}" class="{command_class}"/>',
            f'<text class="label" x="{left + 32.0:.2f}" y="{baseline:.2f}">commanded</text>',
            f'<line x1="{left + 135.0:.2f}" y1="{baseline - 4.0:.2f}" '
            f'x2="{left + 160.0:.2f}" y2="{baseline - 4.0:.2f}" class="{measured_class}"/>',
            f'<text class="label" x="{left + 167.0:.2f}" y="{baseline:.2f}">measured</text>',
        ]
    )


def _relative_times(samples: list[TrackingSample]) -> NDArray[np.float64]:
    start = samples[0].elapsed_s
    return np.array([sample.elapsed_s - start for sample in samples])


def _append_segment_separators(
    elements: list[str],
    panel: tuple[float, float, float, float],
    times: NDArray[np.float64],
    time_bounds: tuple[float, float],
    samples: list[TrackingSample],
) -> None:
    for index in range(1, len(samples)):
        if samples[index].segment == samples[index - 1].segment:
            continue
        x = _map_x(float(times[index]), panel, time_bounds)
        elements.append(
            f'<line class="separator" x1="{x:.2f}" y1="{panel[1]:.2f}" '
            f'x2="{x:.2f}" y2="{panel[1] + panel[3]:.2f}"/>'
        )


def _decimate(samples: list[TrackingSample], maximum: int) -> list[TrackingSample]:
    if len(samples) <= maximum:
        return samples
    step = math.ceil(len(samples) / maximum)
    result = samples[::step]
    if result[-1] is not samples[-1]:
        result.append(samples[-1])
    return result


def _padded_bounds(
    values: NDArray[np.float64],
    *,
    include_zero: bool = False,
    minimum_span: float | None = None,
) -> tuple[float, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if include_zero:
        minimum = min(0.0, minimum)
        maximum = max(0.0, maximum)
    if minimum_span is not None:
        if not math.isfinite(minimum_span) or minimum_span <= 0.0:
            raise ValueError("minimum span must be positive and finite")
        if maximum - minimum < minimum_span:
            center = (minimum + maximum) / 2.0
            minimum = center - minimum_span / 2.0
            maximum = center + minimum_span / 2.0
    span = maximum - minimum
    padding = (
        span * PLOT_PADDING_FRACTION
        if span > BOUNDS_EPSILON
        else max(abs(maximum) * PLOT_PADDING_FRACTION, MIN_PLOT_PADDING)
    )
    return minimum - padding, maximum + padding


def _equal_aspect_bounds(
    points: NDArray[np.float64], panel_width: float, panel_height: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_bounds = _padded_bounds(points[:, 0])
    y_bounds = _padded_bounds(points[:, 1])
    x_center = sum(x_bounds) / 2.0
    y_center = sum(y_bounds) / 2.0
    x_span = x_bounds[1] - x_bounds[0]
    y_span = y_bounds[1] - y_bounds[0]
    units_per_pixel = max(x_span / panel_width, y_span / panel_height)
    return (
        (
            x_center - units_per_pixel * panel_width / 2.0,
            x_center + units_per_pixel * panel_width / 2.0,
        ),
        (
            y_center - units_per_pixel * panel_height / 2.0,
            y_center + units_per_pixel * panel_height / 2.0,
        ),
    )


def _append_axes(  # noqa: PLR0913 - SVG axes require independent geometry and labels
    elements: list[str],
    panel: tuple[float, float, float, float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    left, top, width, height = panel
    elements.append(
        f'<text class="panel-title" x="{left:.2f}" y="{top - 20:.2f}">{html.escape(title)}</text>'
    )
    for index in range(6):
        fraction = index / 5.0
        x = left + fraction * width
        y = top + fraction * height
        x_value = x_bounds[0] + fraction * (x_bounds[1] - x_bounds[0])
        y_value = y_bounds[1] - fraction * (y_bounds[1] - y_bounds[0])
        elements.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" '
                f'x2="{x:.2f}" y2="{top + height:.2f}"/>',
                f'<text class="tick" text-anchor="middle" x="{x:.2f}" '
                f'y="{top + height + 18:.2f}">{x_value:.3g}</text>',
                f'<line class="grid" x1="{left:.2f}" y1="{y:.2f}" '
                f'x2="{left + width:.2f}" y2="{y:.2f}"/>',
                f'<text class="tick" text-anchor="end" x="{left - 8:.2f}" '
                f'y="{y + 4:.2f}">{y_value:.3g}</text>',
            ]
        )
    elements.extend(
        [
            f'<rect class="axis" fill="none" x="{left:.2f}" y="{top:.2f}" '
            f'width="{width:.2f}" height="{height:.2f}"/>',
            f'<text class="label" text-anchor="middle" x="{left + width / 2.0:.2f}" '
            f'y="{top + height + 40:.2f}">{html.escape(x_label)}</text>',
            f'<text class="label" text-anchor="middle" transform="translate('
            f'{left - 54:.2f} {top + height / 2.0:.2f}) rotate(-90)">{html.escape(y_label)}</text>',
        ]
    )


def _polyline(
    points: NDArray[np.float64],
    panel: tuple[float, float, float, float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    css_class: str,
) -> str:
    coordinates = " ".join(
        f"{_map_x(float(point[0]), panel, x_bounds):.2f},"
        f"{_map_y(float(point[1]), panel, y_bounds):.2f}"
        for point in points
    )
    return f'<polyline class="{css_class}" points="{coordinates}"/>'


def _map_x(
    value: float,
    panel: tuple[float, float, float, float],
    bounds: tuple[float, float],
) -> float:
    return panel[0] + (value - bounds[0]) / (bounds[1] - bounds[0]) * panel[2]


def _map_y(
    value: float,
    panel: tuple[float, float, float, float],
    bounds: tuple[float, float],
) -> float:
    return panel[1] + panel[3] - (value - bounds[0]) / (bounds[1] - bounds[0]) * panel[3]
