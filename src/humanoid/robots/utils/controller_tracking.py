"""Measure controller tracking with Cartesian and gripper trajectories.

The main robot stack must already be running and idle. This diagnostic acquires
the orchestrator's dedicated SYSTEM source, commands the tool relative to its
measured starting pose while cycling the gripper, and returns the orchestrator
to IDLE when it finishes.
"""

import argparse
import csv
import html
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from humanoid.config import ROBOT_CONFIG, ROBOT_CONFIGS
from humanoid.constants import Topic
from humanoid.logger import get_logger
from humanoid.middleware.publisher import Publisher
from humanoid.middleware.subscriber import Subscriber
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.robots.base import Robot
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import (
    RobotConfig,
    RobotJointCommand,
    RobotName,
    RobotState,
    RobotToolCommand,
)
from humanoid.utils.paths import find_data_root

logger = get_logger(__name__)

Plane = Literal["xy", "xz", "yz"]
Segment = Literal["trajectory", "settle"]
GripperBounds = tuple[NDArray[np.float64], NDArray[np.float64]]

PLANE_AXES: dict[Plane, tuple[int, int]] = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}
AXIS_NAMES = ("x", "y", "z")

DEFAULT_WIDTH_METERS = 0.08
DEFAULT_HEIGHT_METERS = 0.04
DEFAULT_PERIOD_SECONDS = 8.0
DEFAULT_CYCLES = 3
DEFAULT_RAMP_SECONDS = 2.0
DEFAULT_COMMAND_RATE_HZ = 10.0
DEFAULT_SETTLE_SECONDS = 2.0
DEFAULT_START_DELAY_SECONDS = 3.0
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 5.0
DEFAULT_FEEDBACK_TIMEOUT_SECONDS = 1.0
DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION = 0.05
MAX_GRIPPER_LIMIT_MARGIN_FRACTION = 0.5
MAX_PLOT_POINTS = 2_000
BOUNDS_EPSILON = 1e-12
PLOT_PADDING_FRACTION = 0.08
MIN_PLOT_PADDING = 1e-3


@dataclass(frozen=True, kw_only=True)
class ControllerTrackingSettings:
    """Validated parameters for one controller-tracking run."""

    plane: Plane = "xz"
    width_m: float = DEFAULT_WIDTH_METERS
    height_m: float = DEFAULT_HEIGHT_METERS
    period_s: float = DEFAULT_PERIOD_SECONDS
    cycles: int = DEFAULT_CYCLES
    ramp_s: float = DEFAULT_RAMP_SECONDS
    rate_hz: float = DEFAULT_COMMAND_RATE_HZ
    settle_s: float = DEFAULT_SETTLE_SECONDS
    start_delay_s: float = DEFAULT_START_DELAY_SECONDS
    connection_timeout_s: float = DEFAULT_CONNECTION_TIMEOUT_SECONDS
    feedback_timeout_s: float = DEFAULT_FEEDBACK_TIMEOUT_SECONDS
    move_gripper: bool = True
    gripper_min_rad: float | None = None
    gripper_max_rad: float | None = None
    gripper_period_s: float | None = None
    gripper_limit_margin_fraction: float = DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION

    def __post_init__(self) -> None:
        positive_values = {
            "width": self.width_m,
            "height": self.height_m,
            "period": self.period_s,
            "rate": self.rate_hz,
            "connection timeout": self.connection_timeout_s,
            "feedback timeout": self.feedback_timeout_s,
        }
        for label, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
        if self.cycles <= 0:
            raise ValueError("cycles must be positive")
        for label, value in {
            "ramp": self.ramp_s,
            "settle": self.settle_s,
            "start delay": self.start_delay_s,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be non-negative and finite")
        if self.ramp_s > self.duration_s / 2.0:
            raise ValueError("ramp must not exceed half of the trajectory duration")
        self._validate_gripper_settings()

    def _validate_gripper_settings(self) -> None:
        if (self.gripper_min_rad is None) != (self.gripper_max_rad is None):
            raise ValueError("gripper min and max must be provided together")
        if self.gripper_min_rad is not None:
            assert self.gripper_max_rad is not None
            if not math.isfinite(self.gripper_min_rad) or not math.isfinite(self.gripper_max_rad):
                raise ValueError("gripper min and max must be finite")
            if self.gripper_min_rad >= self.gripper_max_rad:
                raise ValueError("gripper min must be less than gripper max")
        if self.gripper_period_s is not None and (
            not math.isfinite(self.gripper_period_s) or self.gripper_period_s <= 0.0
        ):
            raise ValueError("gripper period must be positive and finite")
        if (
            not math.isfinite(self.gripper_limit_margin_fraction)
            or not 0.0 <= self.gripper_limit_margin_fraction < MAX_GRIPPER_LIMIT_MARGIN_FRACTION
        ):
            raise ValueError("gripper limit margin must be finite and in [0, 0.5)")
        if not self.move_gripper and (
            self.gripper_min_rad is not None or self.gripper_period_s is not None
        ):
            raise ValueError("gripper motion options cannot be used with hold gripper")

    @property
    def duration_s(self) -> float:
        return self.period_s * self.cycles

    @property
    def effective_gripper_period_s(self) -> float:
        """Use one gripper cycle per figure eight unless explicitly overridden."""
        return self.period_s if self.gripper_period_s is None else self.gripper_period_s


@dataclass(frozen=True, kw_only=True)
class TrackingSample:
    """One simultaneous desired-versus-measured controller sample."""

    segment: Segment
    elapsed_s: float
    state_timestamp_s: float
    commanded_position_m: NDArray[np.float64]
    measured_position_m: NDArray[np.float64]
    position_error_m: NDArray[np.float64]
    orientation_error_rad: float
    joint_command_timestamp_s: float
    state_minus_joint_command_s: float
    arm_joint_names: tuple[str, ...]
    commanded_arm_joint_positions_rad: NDArray[np.float64]
    measured_arm_joint_positions_rad: NDArray[np.float64]
    arm_joint_position_errors_rad: NDArray[np.float64]
    commanded_gripper_positions_rad: NDArray[np.float64] | None = None
    measured_gripper_positions_rad: NDArray[np.float64] | None = None
    gripper_position_errors_rad: NDArray[np.float64] | None = None


@dataclass(frozen=True, kw_only=True)
class ErrorStatistics:
    rms: float
    mean: float
    p95: float
    maximum: float


@dataclass(frozen=True, kw_only=True)
class TrackingStatistics:
    position_m: ErrorStatistics
    orientation_rad: ErrorStatistics
    arm_joint_position_rad: tuple[ErrorStatistics, ...]
    gripper_position_rad: tuple[ErrorStatistics, ...]


@dataclass(frozen=True, kw_only=True)
class TrackingRun:
    samples: list[TrackingSample]
    completed: bool


@dataclass(frozen=True, kw_only=True)
class RuntimeFeedback:
    """Latest controller output and hardware feedback with receipt times."""

    state: RobotState
    joint_command: RobotJointCommand
    last_state_received_s: float
    last_joint_command_received_s: float
    last_mode_received_s: float


def figure_eight_offset(
    elapsed_s: float, settings: ControllerTrackingSettings
) -> NDArray[np.float64]:
    """Return the smoothly ramped Cartesian offset at ``elapsed_s``."""
    elapsed_s = float(np.clip(elapsed_s, 0.0, settings.duration_s))
    phase = 2.0 * np.pi * elapsed_s / settings.period_s
    envelope = _trajectory_envelope(elapsed_s, settings.duration_s, settings.ramp_s)
    first_axis, second_axis = PLANE_AXES[settings.plane]

    offset = np.zeros(3)
    offset[first_axis] = envelope * settings.width_m * 0.5 * np.sin(phase)
    offset[second_axis] = envelope * settings.height_m * 0.5 * np.sin(2.0 * phase)
    return offset


def gripper_sinusoid(
    elapsed_s: float,
    initial_positions_rad: NDArray[np.float64],
    lower_bounds_rad: NDArray[np.float64],
    upper_bounds_rad: NDArray[np.float64],
    settings: ControllerTrackingSettings,
) -> NDArray[np.float64]:
    """Return a smoothly introduced sinusoid between the gripper bounds."""
    elapsed_s = float(np.clip(elapsed_s, 0.0, settings.duration_s))
    phase = 2.0 * np.pi * elapsed_s / settings.effective_gripper_period_s
    normalized_position = 0.5 + 0.5 * np.sin(phase)
    sinusoidal_target = lower_bounds_rad + normalized_position * (
        upper_bounds_rad - lower_bounds_rad
    )
    envelope = _trajectory_envelope(elapsed_s, settings.duration_s, settings.ramp_s)
    return initial_positions_rad + envelope * (sinusoidal_target - initial_positions_rad)


def tracking_statistics(samples: list[TrackingSample]) -> TrackingStatistics:
    """Summarize tool, arm-joint, and gripper errors for non-empty samples."""
    if not samples:
        raise ValueError("Cannot summarize an empty tracking run")
    position_errors = np.array([np.linalg.norm(sample.position_error_m) for sample in samples])
    orientation_errors = np.array([sample.orientation_error_rad for sample in samples])
    _, _, arm_joint_errors = _arm_joint_sample_matrices(samples)
    _, _, gripper_errors = _gripper_sample_matrices(samples)
    return TrackingStatistics(
        position_m=_error_statistics(position_errors),
        orientation_rad=_error_statistics(orientation_errors),
        arm_joint_position_rad=tuple(
            _error_statistics(np.abs(arm_joint_errors[:, index]))
            for index in range(arm_joint_errors.shape[1])
        ),
        gripper_position_rad=tuple(
            _error_statistics(np.abs(gripper_errors[:, index]))
            for index in range(gripper_errors.shape[1])
        ),
    )


def write_tracking_csv(path: Path, samples: list[TrackingSample]) -> None:
    """Write raw tracking samples for subsequent analysis."""
    arm_joint_commands, _, _ = _arm_joint_sample_matrices(samples)
    arm_joint_names = samples[0].arm_joint_names if samples else ()
    arm_joint_count = arm_joint_commands.shape[1]
    _, _, gripper_errors = _gripper_sample_matrices(samples)
    gripper_count = gripper_errors.shape[1]
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
        ]
        for joint_name in arm_joint_names:
            header.extend(
                [
                    f"{joint_name}_controller_command_rad",
                    f"{joint_name}_measured_rad",
                    f"{joint_name}_error_rad",
                ]
            )
        for index in range(gripper_count):
            number = index + 1
            header.extend(
                [
                    f"gripper_{number}_command_rad",
                    f"gripper_{number}_measured_rad",
                    f"gripper_{number}_error_rad",
                ]
            )
        writer.writerow(header)
        for sample in samples:
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
            ]
            for index in range(arm_joint_count):
                row.extend(
                    [
                        sample.commanded_arm_joint_positions_rad[index],
                        sample.measured_arm_joint_positions_rad[index],
                        sample.arm_joint_position_errors_rad[index],
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
                        ]
                    )
            writer.writerow(row)


def write_tracking_svg(  # noqa: PLR0915 - composes the complete standalone SVG report
    path: Path,
    samples: list[TrackingSample],
    settings: ControllerTrackingSettings,
    robot_name: str,
) -> None:
    """Render a dependency-free SVG report with tool, joint, and gripper plots."""
    if not samples:
        raise ValueError("Cannot plot an empty tracking run")

    trajectory_samples = [sample for sample in samples if sample.segment == "trajectory"]
    summary_samples = trajectory_samples or samples
    stats = tracking_statistics(summary_samples)
    plotted = _decimate(samples, MAX_PLOT_POINTS)
    gripper_commands, gripper_measurements, _ = _gripper_sample_matrices(plotted)
    arm_joint_commands, arm_joint_measurements, _ = _arm_joint_sample_matrices(plotted)
    plane_axes = PLANE_AXES[settings.plane]
    horizontal_axis, vertical_axis = plane_axes

    path_panel = (75.0, 105.0, 515.0, 500.0)
    position_panel = (690.0, 105.0, 515.0, 170.0)
    orientation_panel = (690.0, 365.0, 515.0, 150.0)
    gripper_panel = (690.0, 605.0, 515.0, 170.0)
    joint_count = len(stats.arm_joint_position_rad)
    joint_summary_bottom = 980 + 28 * joint_count
    joint_plots_top = joint_summary_bottom + 70
    joint_plot_row_height = 190
    joint_plot_rows = math.ceil(joint_count / 2)
    joint_panels = [
        (
            75.0 if index % 2 == 0 else 690.0,
            float(joint_plots_top + (index // 2) * joint_plot_row_height),
            515.0,
            120.0,
        )
        for index in range(joint_count)
    ]
    canvas_height = max(
        970,
        960 + 28 * len(stats.gripper_position_rad),
        joint_summary_bottom,
        joint_plots_top + joint_plot_rows * joint_plot_row_height,
    )

    command_plane = np.array([sample.commanded_position_m[list(plane_axes)] for sample in plotted])
    measured_plane = np.array([sample.measured_position_m[list(plane_axes)] for sample in plotted])
    plane_values = np.vstack((command_plane, measured_plane))
    x_bounds, y_bounds = _equal_aspect_bounds(plane_values, path_panel[2], path_panel[3])

    times = np.array([sample.elapsed_s for sample in plotted])
    position_errors_mm = np.array(
        [np.linalg.norm(sample.position_error_m) * 1_000.0 for sample in plotted]
    )
    orientation_errors_deg = np.rad2deg([sample.orientation_error_rad for sample in plotted])
    time_bounds = _padded_bounds(times, include_zero=True)
    position_bounds = _padded_bounds(position_errors_mm, include_zero=True)
    orientation_bounds = _padded_bounds(orientation_errors_deg, include_zero=True)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{canvas_height}" '
        f'viewBox="0 0 1280 {canvas_height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172033}",
        ".title{font-size:26px;font-weight:700}.subtitle{font-size:14px;fill:#586174}",
        ".panel-title{font-size:16px;font-weight:650}.axis{stroke:#9ca7b8;stroke-width:1}",
        ".grid{stroke:#dfe4eb;stroke-width:1}.tick{font-size:11px;fill:#677286}",
        ".label{font-size:12px;fill:#3e485a}.summary{font-size:14px}",
        ".command,.joint-command{fill:none;stroke:#2667d8;stroke-width:2.3}",
        ".measured,.joint-measured{fill:none;stroke:#e07a2d;stroke-width:2.1}",
        ".joint-measured{stroke-width:1.8;stroke-dasharray:5 3}",
        ".error{fill:none;stroke:#a33bc1;stroke-width:2.1}",
        ".separator{stroke:#8b95a6;stroke-width:1;stroke-dasharray:5 5}",
        "</style>",
        f'<rect width="1280" height="{canvas_height}" fill="#f7f9fc"/>',
        '<text class="title" x="40" y="44">Controller tracking measurement</text>',
        (
            f'<text class="subtitle" x="40" y="70">{html.escape(robot_name)} · '
            f"{settings.plane.upper()} plane · {settings.rate_hz:g} Hz · "
            f"{settings.cycles} cycles at {settings.period_s:g} s</text>"
        ),
    ]

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
            '<line x1="300" y1="84" x2="325" y2="84" class="command"/>',
            '<text class="label" x="332" y="88">commanded</text>',
            '<line x1="435" y1="84" x2="460" y2="84" class="measured"/>',
            '<text class="label" x="467" y="88">measured</text>',
        ]
    )

    _append_axes(
        elements,
        position_panel,
        time_bounds,
        position_bounds,
        "elapsed time (s)",
        "position error (mm)",
        "Translation error",
    )
    position_points = np.column_stack((times, position_errors_mm))
    elements.append(
        _polyline(position_points, position_panel, time_bounds, position_bounds, "error")
    )

    _append_axes(
        elements,
        orientation_panel,
        time_bounds,
        orientation_bounds,
        "elapsed time (s)",
        "orientation error (deg)",
        "Orientation error",
    )
    orientation_points = np.column_stack((times, orientation_errors_deg))
    elements.append(
        _polyline(orientation_points, orientation_panel, time_bounds, orientation_bounds, "error")
    )

    if gripper_commands.shape[1]:
        gripper_commands_deg = np.rad2deg(gripper_commands)
        gripper_measurements_deg = np.rad2deg(gripper_measurements)
        gripper_bounds = _padded_bounds(
            np.concatenate((gripper_commands_deg.ravel(), gripper_measurements_deg.ravel()))
        )
        _append_axes(
            elements,
            gripper_panel,
            time_bounds,
            gripper_bounds,
            "elapsed time (s)",
            "joint position (deg)",
            "Gripper command and measurement",
        )
        for index in range(gripper_commands.shape[1]):
            command_points = np.column_stack((times, gripper_commands_deg[:, index]))
            measured_points = np.column_stack((times, gripper_measurements_deg[:, index]))
            elements.extend(
                [
                    _polyline(
                        command_points, gripper_panel, time_bounds, gripper_bounds, "command"
                    ),
                    _polyline(
                        measured_points, gripper_panel, time_bounds, gripper_bounds, "measured"
                    ),
                ]
            )
    else:
        left, top, width, height = gripper_panel
        elements.extend(
            [
                f'<text class="panel-title" x="{left:.2f}" y="{top - 20:.2f}">Gripper</text>',
                f'<rect class="axis" fill="none" x="{left:.2f}" y="{top:.2f}" '
                f'width="{width:.2f}" height="{height:.2f}"/>',
                f'<text class="subtitle" text-anchor="middle" x="{left + width / 2.0:.2f}" '
                f'y="{top + height / 2.0:.2f}">No gripper joints configured</text>',
            ]
        )

    if settings.settle_s > 0.0:
        for panel in (position_panel, orientation_panel, gripper_panel):
            x = _map_x(settings.duration_s, panel, time_bounds)
            elements.append(
                f'<line class="separator" x1="{x:.2f}" y1="{panel[1]:.2f}" '
                f'x2="{x:.2f}" y2="{panel[1] + panel[3]:.2f}"/>'
            )

    position = stats.position_m
    orientation = stats.orientation_rad
    final_error_mm = np.linalg.norm(samples[-1].position_error_m) * 1_000.0
    elements.extend(
        [
            (
                '<text class="summary" x="75" y="850">Trajectory position error: '
                f"RMS {position.rms * 1_000.0:.2f} mm · "
                f"P95 {position.p95 * 1_000.0:.2f} mm · "
                f"max {position.maximum * 1_000.0:.2f} mm</text>"
            ),
            (
                '<text class="summary" x="75" y="878">Trajectory orientation error: '
                f"RMS {np.rad2deg(orientation.rms):.2f}° · "
                f"P95 {np.rad2deg(orientation.p95):.2f}° · "
                f"max {np.rad2deg(orientation.maximum):.2f}°</text>"
            ),
            (
                '<text class="summary" x="75" y="906">'
                f"Final position error after settle: {final_error_mm:.2f} mm</text>"
            ),
        ]
    )
    final_gripper_errors = samples[-1].gripper_position_errors_rad
    for index, gripper_stats in enumerate(stats.gripper_position_rad):
        final_error = 0.0 if final_gripper_errors is None else abs(final_gripper_errors[index])
        elements.append(
            f'<text class="summary" x="690" y="{850 + 28 * index}">'
            f"Gripper {index + 1} error: RMS {np.rad2deg(gripper_stats.rms):.2f}° · "
            f"P95 {np.rad2deg(gripper_stats.p95):.2f}° · "
            f"max {np.rad2deg(gripper_stats.maximum):.2f}° · "
            f"final {np.rad2deg(final_error):.2f}°</text>"
        )
    if stats.arm_joint_position_rad:
        elements.extend(
            [
                (
                    '<text class="panel-title" x="75" y="950">'
                    "Controller output vs measured joints</text>"
                ),
                '<line x1="690" y1="946" x2="715" y2="946" class="joint-command"/>',
                '<text class="label" x="722" y="950">commanded</text>',
                '<line x1="825" y1="946" x2="850" y2="946" class="joint-measured"/>',
                '<text class="label" x="857" y="950">measured</text>',
            ]
        )
    for index, (joint_name, joint_stats) in enumerate(
        zip(samples[0].arm_joint_names, stats.arm_joint_position_rad, strict=True)
    ):
        elements.append(
            f'<text class="summary" x="75" y="{980 + 28 * index}">'
            f"{html.escape(joint_name)} error: RMS {np.rad2deg(joint_stats.rms):.2f}° · "
            f"P95 {np.rad2deg(joint_stats.p95):.2f}° · "
            f"max {np.rad2deg(joint_stats.maximum):.2f}°</text>"
        )

    for index, (joint_name, panel) in enumerate(
        zip(samples[0].arm_joint_names, joint_panels, strict=True)
    ):
        joint_commands_deg = np.rad2deg(arm_joint_commands[:, index])
        joint_measurements_deg = np.rad2deg(arm_joint_measurements[:, index])
        joint_bounds = _padded_bounds(np.concatenate((joint_commands_deg, joint_measurements_deg)))
        _append_axes(
            elements,
            panel,
            time_bounds,
            joint_bounds,
            "elapsed time (s)",
            "joint position (deg)",
            joint_name,
        )
        elements.extend(
            [
                _polyline(
                    np.column_stack((times, joint_commands_deg)),
                    panel,
                    time_bounds,
                    joint_bounds,
                    "joint-command",
                ),
                _polyline(
                    np.column_stack((times, joint_measurements_deg)),
                    panel,
                    time_bounds,
                    joint_bounds,
                    "joint-measured",
                ),
            ]
        )
        if settings.settle_s > 0.0:
            x = _map_x(settings.duration_s, panel, time_bounds)
            elements.append(
                f'<line class="separator" x1="{x:.2f}" y1="{panel[1]:.2f}" '
                f'x2="{x:.2f}" y2="{panel[1] + panel[3]:.2f}"/>'
            )
    elements.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def run_controller_tracking(  # noqa: PLR0915 - owns the utility's complete safety lifecycle
    settings: ControllerTrackingSettings,
    robot_config: RobotConfig = ROBOT_CONFIG,
) -> TrackingRun:
    """Acquire SYSTEM mode, execute the motion, and collect tracking samples."""
    robot = Robot(robot_config)
    publisher = Publisher()
    subscriber = Subscriber(
        topics=[Topic.ROBOT_STATE, Topic.ROBOT_JOINT_COMMAND, Topic.ORCHESTRATOR_MODE]
    )
    orchestrator = OrchestratorClient(publisher=publisher)
    system_requested = False
    samples: list[TrackingSample] = []
    completed = False

    try:
        initial_mode = subscriber.receive(
            Topic.ORCHESTRATOR_MODE,
            timeout=round(settings.connection_timeout_s * 1_000.0),
        )
        if initial_mode is None:
            raise RuntimeError("Timed out waiting for the orchestrator; start the main stack first")
        if initial_mode.mode is not Mode.IDLE:
            raise RuntimeError(
                f"The orchestrator must be idle before this test; current mode is "
                f"{initial_mode.mode.value}"
            )

        initial_state = subscriber.receive(
            Topic.ROBOT_STATE,
            timeout=round(settings.connection_timeout_s * 1_000.0),
        )
        if initial_state is None:
            raise RuntimeError("Timed out waiting for robot state; start the main stack first")
        if initial_state.joint_positions.shape != (robot.model.nq,):
            raise RuntimeError(
                f"Robot state has {len(initial_state.joint_positions)} positions, but "
                f"{robot_config.name.value} expects {robot.model.nq}; check --robot"
            )

        anchor_pose = robot.get_tool_command_pose(initial_state.joint_positions)
        gripper_indices = robot.get_gripper_position_indices()
        initial_gripper_positions = (
            initial_state.joint_positions[gripper_indices].copy() if gripper_indices else None
        )
        gripper_bounds = _resolve_gripper_bounds(robot, settings)
        _log_run_plan(
            settings,
            robot_config,
            anchor_pose,
            initial_gripper_positions,
            gripper_bounds,
        )
        _count_down(settings.start_delay_s)
        _wait_for_mode(subscriber, Mode.IDLE, min(settings.connection_timeout_s, 1.0))

        system_requested = True
        system_requested_at = time.perf_counter()
        orchestrator.request_system()
        _wait_for_mode(subscriber, Mode.SYSTEM, settings.connection_timeout_s)

        initial_joint_command = _wait_for_fresh_joint_command(
            subscriber,
            minimum_timestamp_s=system_requested_at,
            timeout_s=settings.connection_timeout_s,
        )
        if initial_joint_command.joint_positions.shape != (robot.model.nq,):
            raise RuntimeError(
                f"Controller command has {len(initial_joint_command.joint_positions)} positions, "
                f"but {robot_config.name.value} expects {robot.model.nq}; check --robot"
            )

        feedback_received_at = time.monotonic()
        feedback = RuntimeFeedback(
            state=initial_state,
            joint_command=initial_joint_command,
            last_state_received_s=feedback_received_at,
            last_joint_command_received_s=feedback_received_at,
            last_mode_received_s=feedback_received_at,
        )
        start = time.monotonic()
        next_tick = start

        while True:
            now = time.monotonic()
            elapsed_s = min(now - start, settings.duration_s)
            feedback = _refresh_runtime_state(subscriber, feedback, settings.feedback_timeout_s)
            commanded_pose = pin.SE3(
                anchor_pose.rotation.copy(),
                anchor_pose.translation + figure_eight_offset(elapsed_s, settings),
            )
            commanded_gripper_positions = _commanded_gripper_positions(
                elapsed_s,
                initial_gripper_positions,
                gripper_bounds,
                settings,
            )
            command = RobotToolCommand(
                timestamp=time.time(),
                pose=commanded_pose,
                gripper_positions=commanded_gripper_positions,
            )
            publisher.publish(command, topic=Topic.SYSTEM_TOOL_COMMAND)
            samples.append(
                _tracking_sample(
                    "trajectory",
                    elapsed_s,
                    command,
                    feedback,
                    robot,
                )
            )
            if elapsed_s >= settings.duration_s:
                completed = True
                break
            next_tick = _sleep_until_next_tick(next_tick, settings.rate_hz)

        settle_end = time.monotonic() + settings.settle_s
        while time.monotonic() < settle_end:
            feedback = _refresh_runtime_state(subscriber, feedback, settings.feedback_timeout_s)
            command = RobotToolCommand(
                timestamp=time.time(),
                pose=anchor_pose,
                gripper_positions=initial_gripper_positions,
            )
            publisher.publish(command, topic=Topic.SYSTEM_TOOL_COMMAND)
            samples.append(
                _tracking_sample(
                    "settle",
                    time.monotonic() - start,
                    command,
                    feedback,
                    robot,
                )
            )
            next_tick = _sleep_until_next_tick(next_tick, settings.rate_hz)
    except KeyboardInterrupt:
        logger.warning("Controller tracking run interrupted; returning the controller to idle")
    finally:
        if system_requested:
            orchestrator.request_idle()
        subscriber.close()

    return TrackingRun(samples=samples, completed=completed)


def _trajectory_envelope(elapsed_s: float, duration_s: float, ramp_s: float) -> float:
    if ramp_s == 0.0:
        return 1.0
    if elapsed_s < ramp_s:
        return _smootherstep(elapsed_s / ramp_s)
    if elapsed_s > duration_s - ramp_s:
        return _smootherstep((duration_s - elapsed_s) / ramp_s)
    return 1.0


def _resolve_gripper_bounds(
    robot: Robot,
    settings: ControllerTrackingSettings,
) -> GripperBounds | None:
    limits = robot.get_gripper_limits()
    if not settings.move_gripper or not limits:
        return None

    model_bounds = np.asarray(limits, dtype=float)
    model_lower = model_bounds[:, 0]
    model_upper = model_bounds[:, 1]
    if not np.all(np.isfinite(model_bounds)) or np.any(model_lower >= model_upper):
        raise RuntimeError(f"The robot model has invalid gripper limits: {limits}")
    if settings.gripper_min_rad is not None:
        assert settings.gripper_max_rad is not None
        lower = np.full(len(limits), settings.gripper_min_rad)
        upper = np.full(len(limits), settings.gripper_max_rad)
        if np.any(lower < model_lower - BOUNDS_EPSILON) or np.any(
            upper > model_upper + BOUNDS_EPSILON
        ):
            raise RuntimeError(
                "Requested gripper bounds are outside the robot model limits: "
                f"requested [{settings.gripper_min_rad:g}, {settings.gripper_max_rad:g}] rad, "
                f"model {limits}"
            )
    else:
        margin = settings.gripper_limit_margin_fraction * (model_upper - model_lower)
        lower = model_lower + margin
        upper = model_upper - margin

    if np.any(lower >= upper):
        raise RuntimeError("The configured gripper limits do not contain a usable range")
    return lower, upper


def _commanded_gripper_positions(
    elapsed_s: float,
    initial_positions_rad: NDArray[np.float64] | None,
    bounds: GripperBounds | None,
    settings: ControllerTrackingSettings,
) -> NDArray[np.float64] | None:
    if initial_positions_rad is None:
        return None
    if bounds is None:
        return initial_positions_rad.copy()
    lower, upper = bounds
    return gripper_sinusoid(elapsed_s, initial_positions_rad, lower, upper, settings)


def _smootherstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _error_statistics(values: NDArray[np.float64]) -> ErrorStatistics:
    return ErrorStatistics(
        rms=float(np.sqrt(np.mean(np.square(values)))),
        mean=float(np.mean(values)),
        p95=float(np.percentile(values, 95)),
        maximum=float(np.max(values)),
    )


def _gripper_sample_matrices(
    samples: list[TrackingSample],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if not samples:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty.copy(), empty.copy()

    first_values = (
        samples[0].commanded_gripper_positions_rad,
        samples[0].measured_gripper_positions_rad,
        samples[0].gripper_position_errors_rad,
    )
    if all(values is None for values in first_values):
        if any(
            values is not None
            for sample in samples
            for values in (
                sample.commanded_gripper_positions_rad,
                sample.measured_gripper_positions_rad,
                sample.gripper_position_errors_rad,
            )
        ):
            raise ValueError("Gripper data must be present for every tracking sample or none")
        empty = np.empty((len(samples), 0), dtype=float)
        return empty, empty.copy(), empty.copy()
    if any(values is None for values in first_values):
        raise ValueError("Each gripper sample must contain command, measurement, and error")

    assert first_values[0] is not None
    expected_shape = first_values[0].shape
    if len(expected_shape) != 1:
        raise ValueError("Gripper tracking values must be one-dimensional")

    fields: list[list[NDArray[np.float64]]] = [[], [], []]
    for sample in samples:
        sample_values = (
            sample.commanded_gripper_positions_rad,
            sample.measured_gripper_positions_rad,
            sample.gripper_position_errors_rad,
        )
        if any(values is None or values.shape != expected_shape for values in sample_values):
            raise ValueError("Gripper tracking values must have consistent shapes")
        for field, values in zip(fields, sample_values, strict=True):
            assert values is not None
            field.append(values)
    return np.vstack(fields[0]), np.vstack(fields[1]), np.vstack(fields[2])


def _arm_joint_sample_matrices(
    samples: list[TrackingSample],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return consistent controller-command, measurement, and error matrices."""
    if not samples:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty.copy(), empty.copy()

    expected_names = samples[0].arm_joint_names
    expected_shape = (len(expected_names),)
    fields: list[list[NDArray[np.float64]]] = [[], [], []]
    for sample in samples:
        if sample.arm_joint_names != expected_names:
            raise ValueError("Arm joint names must be consistent across tracking samples")
        sample_values = (
            sample.commanded_arm_joint_positions_rad,
            sample.measured_arm_joint_positions_rad,
            sample.arm_joint_position_errors_rad,
        )
        if any(values.shape != expected_shape for values in sample_values):
            raise ValueError("Arm joint tracking values must match the configured joint names")
        for field, values in zip(fields, sample_values, strict=True):
            field.append(values)
    return np.vstack(fields[0]), np.vstack(fields[1]), np.vstack(fields[2])


def _tracking_sample(
    segment: Segment,
    elapsed_s: float,
    command: RobotToolCommand,
    feedback: RuntimeFeedback,
    robot: Robot,
) -> TrackingSample:
    robot_state = feedback.state
    joint_command = feedback.joint_command
    measured_pose = robot.get_tool_command_pose(robot_state.joint_positions)
    position_error = command.pose.translation - measured_pose.translation
    orientation_error = pin.log3(measured_pose.rotation.T @ command.pose.rotation)
    gripper_indices = robot.get_gripper_position_indices()
    measured_gripper_positions_rad = (
        robot_state.joint_positions[gripper_indices].copy()
        if command.gripper_positions is not None
        else None
    )
    gripper_position_errors_rad = (
        command.gripper_positions - measured_gripper_positions_rad
        if command.gripper_positions is not None and measured_gripper_positions_rad is not None
        else None
    )
    arm_joint_indices = robot.get_arm_joint_indices()
    arm_joint_names = tuple(robot.joint_idx_to_name(index) for index in arm_joint_indices)
    commanded_arm_joint_positions_rad = np.array(
        [
            robot.joint_position_from_q(joint_command.joint_positions, index)
            for index in arm_joint_indices
        ]
    )
    measured_arm_joint_positions_rad = np.array(
        [
            robot.joint_position_from_q(robot_state.joint_positions, index)
            for index in arm_joint_indices
        ]
    )
    raw_arm_joint_errors = commanded_arm_joint_positions_rad - measured_arm_joint_positions_rad
    arm_joint_position_errors_rad = np.arctan2(
        np.sin(raw_arm_joint_errors), np.cos(raw_arm_joint_errors)
    )
    return TrackingSample(
        segment=segment,
        elapsed_s=elapsed_s,
        state_timestamp_s=robot_state.timestamp,
        commanded_position_m=command.pose.translation.copy(),
        measured_position_m=measured_pose.translation.copy(),
        position_error_m=position_error,
        orientation_error_rad=float(np.linalg.norm(orientation_error)),
        joint_command_timestamp_s=joint_command.timestamp,
        state_minus_joint_command_s=robot_state.timestamp - joint_command.timestamp,
        arm_joint_names=arm_joint_names,
        commanded_arm_joint_positions_rad=commanded_arm_joint_positions_rad,
        measured_arm_joint_positions_rad=measured_arm_joint_positions_rad,
        arm_joint_position_errors_rad=arm_joint_position_errors_rad,
        commanded_gripper_positions_rad=(
            command.gripper_positions.copy() if command.gripper_positions is not None else None
        ),
        measured_gripper_positions_rad=measured_gripper_positions_rad,
        gripper_position_errors_rad=gripper_position_errors_rad,
    )


def _wait_for_mode(subscriber: Subscriber, expected: Mode, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_mode: Mode | None = None
    while time.monotonic() < deadline:
        remaining_ms = max(1, round((deadline - time.monotonic()) * 1_000.0))
        message = subscriber.receive(Topic.ORCHESTRATOR_MODE, timeout=min(remaining_ms, 100))
        if message is None:
            continue
        last_mode = message.mode
        if last_mode is expected:
            return
    observed = "no mode" if last_mode is None else last_mode.value
    raise RuntimeError(
        f"Timed out waiting for orchestrator mode {expected.value}; last observed {observed}"
    )


def _wait_for_fresh_joint_command(
    subscriber: Subscriber,
    minimum_timestamp_s: float,
    timeout_s: float,
) -> RobotJointCommand:
    """Wait for a controller output created after SYSTEM mode was requested."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining_ms = max(1, round((deadline - time.monotonic()) * 1_000.0))
        command = subscriber.receive(Topic.ROBOT_JOINT_COMMAND, timeout=min(remaining_ms, 100))
        if command is not None and command.timestamp >= minimum_timestamp_s:
            return command
    raise RuntimeError("Timed out waiting for a fresh controller joint command")


def _refresh_runtime_state(
    subscriber: Subscriber,
    feedback: RuntimeFeedback,
    timeout_s: float,
) -> RuntimeFeedback:
    now = time.monotonic()
    latest_state = feedback.state
    latest_joint_command = feedback.joint_command
    last_state_received = feedback.last_state_received_s
    last_joint_command_received = feedback.last_joint_command_received_s
    last_mode_received = feedback.last_mode_received_s
    state = subscriber.receive(Topic.ROBOT_STATE)
    if state is not None:
        latest_state = state
        last_state_received = now

    joint_command = subscriber.receive(Topic.ROBOT_JOINT_COMMAND)
    if joint_command is not None:
        latest_joint_command = joint_command
        last_joint_command_received = now

    mode_message = subscriber.receive(Topic.ORCHESTRATOR_MODE)
    if mode_message is not None:
        if mode_message.mode is not Mode.SYSTEM:
            raise RuntimeError(
                f"Orchestrator left system mode for {mode_message.mode.value}; stopping test"
            )
        last_mode_received = now

    if now - last_state_received > timeout_s:
        raise RuntimeError(f"Robot-state feedback was stale for more than {timeout_s:g} seconds")
    if now - last_joint_command_received > timeout_s:
        raise RuntimeError(
            f"Controller joint commands were stale for more than {timeout_s:g} seconds"
        )
    if now - last_mode_received > timeout_s:
        raise RuntimeError(f"Orchestrator mode was stale for more than {timeout_s:g} seconds")
    return RuntimeFeedback(
        state=latest_state,
        joint_command=latest_joint_command,
        last_state_received_s=last_state_received,
        last_joint_command_received_s=last_joint_command_received,
        last_mode_received_s=last_mode_received,
    )


def _sleep_until_next_tick(previous_tick: float, rate_hz: float) -> float:
    period_s = 1.0 / rate_hz
    next_tick = previous_tick + period_s
    now = time.monotonic()
    if next_tick < now:
        next_tick = now
    else:
        time.sleep(next_tick - now)
    return next_tick


def _count_down(delay_s: float) -> None:
    if delay_s == 0.0:
        return
    logger.warning("Motion begins in %.1f seconds; press Ctrl-C to cancel", delay_s)
    deadline = time.monotonic() + delay_s
    while (remaining := deadline - time.monotonic()) > 0.0:
        time.sleep(min(remaining, 0.1))


def _log_run_plan(
    settings: ControllerTrackingSettings,
    robot_config: RobotConfig,
    anchor_pose: pin.SE3,
    initial_gripper_positions_rad: NDArray[np.float64] | None,
    gripper_bounds_rad: GripperBounds | None,
) -> None:
    logger.info(
        "Controller tracking: robot=%s, plane=%s, width=%.1f mm, height=%.1f mm, "
        "period=%.1f s, cycles=%d, rate=%.1f Hz",
        robot_config.name,
        settings.plane,
        settings.width_m * 1_000.0,
        settings.height_m * 1_000.0,
        settings.period_s,
        settings.cycles,
        settings.rate_hz,
    )
    logger.info("Measured tool anchor: %s m", np.array2string(anchor_pose.translation, precision=4))
    if initial_gripper_positions_rad is None:
        logger.info("No gripper joints are configured; measuring tool tracking only")
    elif gripper_bounds_rad is None:
        logger.info(
            "Holding measured gripper position: %s rad",
            np.array2string(initial_gripper_positions_rad, precision=4),
        )
    else:
        lower, upper = gripper_bounds_rad
        logger.info(
            "Gripper sinusoid: lower=%s rad, upper=%s rad, period=%.1f s",
            np.array2string(lower, precision=4),
            np.array2string(upper, precision=4),
            settings.effective_gripper_period_s,
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
    values: NDArray[np.float64], *, include_zero: bool = False
) -> tuple[float, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if include_zero:
        minimum = min(0.0, minimum)
        maximum = max(0.0, maximum)
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


def _default_output_directory() -> Path:
    return find_data_root(__file__) / "logs" / "tracking"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Command a Cartesian figure eight plus a gripper sinusoid and report tracking error."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--plane", choices=tuple(PLANE_AXES), default="xz")
    parser.add_argument(
        "--robot",
        type=RobotName,
        choices=list(RobotName),
        default=ROBOT_CONFIG.name,
        help="Robot model; defaults to HUMANOID_ROBOT or the project default",
    )
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH_METERS, help="Width in meters")
    parser.add_argument(
        "--height", type=float, default=DEFAULT_HEIGHT_METERS, help="Height in meters"
    )
    parser.add_argument("--period", type=float, default=DEFAULT_PERIOD_SECONDS)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--ramp", type=float, default=DEFAULT_RAMP_SECONDS)
    parser.add_argument("--rate", type=float, default=DEFAULT_COMMAND_RATE_HZ)
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument("--start-delay", type=float, default=DEFAULT_START_DELAY_SECONDS)
    parser.add_argument(
        "--connection-timeout", type=float, default=DEFAULT_CONNECTION_TIMEOUT_SECONDS
    )
    parser.add_argument("--feedback-timeout", type=float, default=DEFAULT_FEEDBACK_TIMEOUT_SECONDS)
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
        help="Gripper sinusoid period in seconds; defaults to the figure-eight period",
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
        ramp_s=args.ramp,
        rate_hz=args.rate,
        settle_s=args.settle,
        start_delay_s=args.start_delay,
        connection_timeout_s=args.connection_timeout,
        feedback_timeout_s=args.feedback_timeout,
        move_gripper=not args.hold_gripper,
        gripper_min_rad=args.gripper_min,
        gripper_max_rad=args.gripper_max,
        gripper_period_s=args.gripper_period,
        gripper_limit_margin_fraction=args.gripper_limit_margin,
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
    svg_path = args.output_dir / f"{stem}.svg"
    write_tracking_csv(csv_path, run.samples)
    write_tracking_svg(svg_path, run.samples, settings, robot_config.name.value)

    trajectory_samples = [sample for sample in run.samples if sample.segment == "trajectory"]
    stats = tracking_statistics(trajectory_samples or run.samples)
    final_error_mm = np.linalg.norm(run.samples[-1].position_error_m) * 1_000.0
    logger.info(
        "Position error: RMS %.2f mm, mean %.2f mm, P95 %.2f mm, max %.2f mm",
        stats.position_m.rms * 1_000.0,
        stats.position_m.mean * 1_000.0,
        stats.position_m.p95 * 1_000.0,
        stats.position_m.maximum * 1_000.0,
    )
    logger.info(
        "Orientation error: RMS %.2f deg, P95 %.2f deg, max %.2f deg",
        np.rad2deg(stats.orientation_rad.rms),
        np.rad2deg(stats.orientation_rad.p95),
        np.rad2deg(stats.orientation_rad.maximum),
    )
    for joint_name, joint_stats in zip(
        run.samples[0].arm_joint_names, stats.arm_joint_position_rad, strict=True
    ):
        logger.info(
            "%s controller-vs-measured error: RMS %.2f deg, P95 %.2f deg, max %.2f deg",
            joint_name,
            np.rad2deg(joint_stats.rms),
            np.rad2deg(joint_stats.p95),
            np.rad2deg(joint_stats.maximum),
        )
    command_to_state_ms = np.array(
        [sample.state_minus_joint_command_s * 1_000.0 for sample in run.samples]
    )
    logger.info(
        "State minus controller-command timestamp: median %.1f ms, range [%.1f, %.1f] ms",
        np.median(command_to_state_ms),
        np.min(command_to_state_ms),
        np.max(command_to_state_ms),
    )
    logger.info("Final position error after settle: %.2f mm", final_error_mm)
    final_gripper_errors = run.samples[-1].gripper_position_errors_rad
    for index, gripper_stats in enumerate(stats.gripper_position_rad):
        final_error = 0.0 if final_gripper_errors is None else abs(final_gripper_errors[index])
        logger.info(
            "Gripper %d error: RMS %.2f deg, mean %.2f deg, P95 %.2f deg, "
            "max %.2f deg, final %.2f deg",
            index + 1,
            np.rad2deg(gripper_stats.rms),
            np.rad2deg(gripper_stats.mean),
            np.rad2deg(gripper_stats.p95),
            np.rad2deg(gripper_stats.maximum),
            np.rad2deg(final_error),
        )
    logger.info("Raw samples: %s", csv_path)
    logger.info("Plot: %s", svg_path)
    if not run.completed:
        logger.warning("Report contains a partial run because the test was interrupted")


if __name__ == "__main__":
    main()
