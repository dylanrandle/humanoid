import csv
import json

import numpy as np
import pinocchio as pin
import pytest

import humanoid.robots.utils.controller_tracking.cli as cli_module
from humanoid.robots.utils.controller_tracking import (
    DEFAULT_COMMAND_RATE_HZ,
    DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION,
    DEFAULT_GRIPPER_PERIOD_SECONDS,
    FIGURE_EIGHT_SETTINGS,
    ControllerCommandTiming,
    ControllerTrackingSettings,
    Segment,
    TrackingSample,
    controller_publication_statistics,
    figure_eight_offset,
    figure_eight_pose,
    figure_eight_velocity,
    gripper_sinusoid,
    tracking_statistics,
    write_controller_timing_csv,
    write_run_metrics_json,
    write_tracking_csv,
    write_tracking_plots,
)
from humanoid.robots.utils.controller_tracking.metadata import write_run_comparison
from humanoid.robots.utils.controller_tracking.models import (
    FIGURE_EIGHT_LOOPS_PER_SETTING,
    FIGURE_EIGHT_TOTAL_LOOPS,
    FigureEightSetting,
)
from humanoid.robots.utils.controller_tracking.smoothness import analyze_smoothness
from humanoid.types.actuator import ActuatorEffortLimits
from humanoid.types.controller_tracking import ActuatorEffortTrace, NativeJointSample
from humanoid.types.robot import CartesianVelocity

EXPECTED_TIMING_COMMAND_COUNT = 4
EXPECTED_LOOPS_PER_SETTING = 2
EXPECTED_TOTAL_LOOPS = 18
EXPECTED_GRIPPER_CYCLES = 10
MINIMUM_LIMIT_OCCUPANCY = 0.9


def _sample(  # noqa: PLR0913 - test builder exposes independent sample dimensions
    elapsed_s: float,
    osc_position_error_m: np.ndarray,
    osc_orientation_error_rad: float,
    *,
    segment: Segment = "figure_eight",
    setting: str = "xz_1x",
    end_to_end_position_error_m: np.ndarray | None = None,
    end_to_end_orientation_error_rad: float | None = None,
    arm_joint_positions_rad: tuple[np.ndarray, np.ndarray] | None = None,
    gripper_positions_rad: tuple[np.ndarray, np.ndarray] | None = None,
    commanded_velocity: CartesianVelocity | None = None,
) -> TrackingSample:
    reference = np.array([0.1 + 0.01 * elapsed_s, 0.2, 0.3])
    osc = reference - osc_position_error_m
    if end_to_end_position_error_m is None:
        end_to_end_position_error_m = 2.0 * osc_position_error_m
    if end_to_end_orientation_error_rad is None:
        end_to_end_orientation_error_rad = 2.0 * osc_orientation_error_rad
    commanded_gripper, measured_gripper = (
        gripper_positions_rad if gripper_positions_rad is not None else (None, None)
    )
    gripper_error = (
        commanded_gripper - measured_gripper
        if commanded_gripper is not None and measured_gripper is not None
        else None
    )
    gripper_names = ("gripper_1",) if commanded_gripper is not None else ()
    commanded_arm, measured_arm = (
        arm_joint_positions_rad
        if arm_joint_positions_rad is not None
        else (np.empty(0), np.empty(0))
    )
    arm_error = np.arctan2(
        np.sin(commanded_arm - measured_arm),
        np.cos(commanded_arm - measured_arm),
    )
    return TrackingSample(
        segment=segment,
        setting=setting,
        elapsed_s=elapsed_s,
        state_timestamp_s=100.0 + elapsed_s,
        reference_position_m=reference,
        osc_position_m=osc,
        measured_position_m=reference - end_to_end_position_error_m,
        osc_position_error_m=osc_position_error_m,
        end_to_end_position_error_m=end_to_end_position_error_m,
        osc_orientation_error_rad=osc_orientation_error_rad,
        end_to_end_orientation_error_rad=end_to_end_orientation_error_rad,
        joint_command_timestamp_s=99.98 + elapsed_s,
        state_minus_joint_command_s=0.02,
        arm_joint_names=("arm_1", "arm_2") if arm_joint_positions_rad is not None else (),
        commanded_arm_joint_positions_rad=commanded_arm,
        measured_arm_joint_positions_rad=measured_arm,
        arm_joint_position_errors_rad=arm_error,
        commanded_arm_joint_velocities_rad_s=None,
        measured_arm_joint_velocities_rad_s=np.zeros_like(measured_arm),
        gripper_joint_names=gripper_names,
        commanded_gripper_positions_rad=commanded_gripper,
        measured_gripper_positions_rad=measured_gripper,
        gripper_position_errors_rad=gripper_error,
        commanded_gripper_velocities_rad_s=None,
        measured_gripper_velocities_rad_s=(
            np.zeros_like(measured_gripper) if measured_gripper is not None else None
        ),
        commanded_linear_velocity_m_s=(
            commanded_velocity.linear if commanded_velocity is not None else None
        ),
        commanded_angular_velocity_rad_s=(
            commanded_velocity.angular if commanded_velocity is not None else None
        ),
    )


def test_defaults_match_controller_rate_and_figure_eight_matrix():
    settings = ControllerTrackingSettings()

    assert settings.rate_hz == DEFAULT_COMMAND_RATE_HZ == pytest.approx(30.0)
    assert FIGURE_EIGHT_LOOPS_PER_SETTING == EXPECTED_LOOPS_PER_SETTING
    assert FIGURE_EIGHT_TOTAL_LOOPS == EXPECTED_TOTAL_LOOPS
    assert settings.duration_s == pytest.approx(162.0)
    assert settings.move_gripper
    assert settings.gripper_period_s == DEFAULT_GRIPPER_PERIOD_SECONDS == pytest.approx(16.0)
    assert settings.gripper_cycle_count == EXPECTED_GRIPPER_CYCLES
    assert settings.effective_gripper_period_s == pytest.approx(16.2)
    assert settings.velocity_feedforward
    assert settings.gripper_limit_margin_fraction == DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION


def test_cli_exposes_figure_eight_orientation_and_size_options():
    destinations = {action.dest for action in cli_module._build_parser()._actions}

    assert {"width", "height", "period", "orientation_bias_deg"} <= destinations


def test_figure_eight_settings_cover_all_planes_and_sizes():
    assert [(setting.plane, setting.size_multiplier) for setting in FIGURE_EIGHT_SETTINGS] == [
        (plane, size) for size in (1.0, 1.5, 2.0) for plane in ("xy", "xz", "yz")
    ]


def test_figure_eight_scales_and_stops_for_each_setting():
    settings = ControllerTrackingSettings(period_s=8.0)
    anchor = pin.SE3.Identity()
    sample_time = 4.0
    small_setting = FigureEightSetting(plane="xz", size_multiplier=1.0)
    large_setting = FigureEightSetting(plane="xz", size_multiplier=2.0)

    small = figure_eight_offset(sample_time, settings, small_setting)
    large = figure_eight_offset(sample_time, settings, large_setting)

    np.testing.assert_allclose(large, 2.0 * small, atol=1e-12)
    for elapsed_s in (0.0, settings.setting_duration_s):
        pose = figure_eight_pose(anchor, elapsed_s, settings, small_setting)
        velocity = figure_eight_velocity(anchor, elapsed_s, settings, small_setting)
        np.testing.assert_allclose(pose.translation, anchor.translation, atol=1e-12)
        np.testing.assert_allclose(pose.rotation, anchor.rotation, atol=1e-12)
        np.testing.assert_allclose(velocity.linear, 0.0, atol=1e-12)
        np.testing.assert_allclose(velocity.angular, 0.0, atol=1e-12)


def test_figure_eight_velocity_matches_pose_derivative():
    settings = ControllerTrackingSettings(period_s=8.0, orientation_bias_rad=0.3)
    setting = FigureEightSetting(plane="xz", size_multiplier=1.5)
    anchor = pin.SE3(pin.exp3(np.array([0.2, -0.1, 0.3])), np.array([0.1, 0.2, 0.3]))
    elapsed_s = 5.0
    epsilon = 1e-5

    before = figure_eight_pose(anchor, elapsed_s - epsilon, settings, setting)
    after = figure_eight_pose(anchor, elapsed_s + epsilon, settings, setting)
    velocity = figure_eight_velocity(anchor, elapsed_s, settings, setting)

    np.testing.assert_allclose(
        velocity.linear,
        (after.translation - before.translation) / (2.0 * epsilon),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        velocity.angular,
        pin.log3(after.rotation @ before.rotation.T) / (2.0 * epsilon),
        rtol=1e-6,
        atol=1e-8,
    )


def test_orientation_bias_tilts_tool_forward_axis_toward_center():
    settings = ControllerTrackingSettings(period_s=8.0, orientation_bias_rad=0.35)
    setting = FigureEightSetting(plane="xy", size_multiplier=1.0)
    anchor = pin.SE3.Identity()
    elapsed_s = 4.0
    pose = figure_eight_pose(anchor, elapsed_s, settings, setting)
    inward = -figure_eight_offset(elapsed_s, settings, setting)
    inward /= np.linalg.norm(inward)
    anchor_forward = anchor.rotation[:, 2]
    commanded_forward = pose.rotation[:, 2]

    assert commanded_forward @ inward > anchor_forward @ inward


def test_gripper_profile_starts_and_finishes_at_current_position_at_rest():
    settings = ControllerTrackingSettings(period_s=2.0, gripper_period_s=4.0)
    initial = np.array([0.3])
    lower = np.array([0.1])
    upper = np.array([1.9])
    epsilon = 1e-3

    np.testing.assert_allclose(gripper_sinusoid(0.0, initial, lower, upper, settings), initial)
    np.testing.assert_allclose(
        gripper_sinusoid(settings.duration_s, initial, lower, upper, settings),
        initial,
    )
    start_rate = (gripper_sinusoid(epsilon, initial, lower, upper, settings) - initial) / epsilon
    end_rate = (
        initial - gripper_sinusoid(settings.duration_s - epsilon, initial, lower, upper, settings)
    ) / epsilon
    np.testing.assert_allclose(start_rate, 0.0, atol=1e-5)
    np.testing.assert_allclose(end_rate, 0.0, atol=1e-5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"period_s": 0.0},
        {"orientation_bias_rad": -0.1},
        {"orientation_bias_rad": np.pi},
        {"settle_s": -1.0},
        {"shake_cutoff_hz": 15.0},
    ],
)
def test_settings_reject_invalid_values(overrides):
    with pytest.raises(ValueError):
        ControllerTrackingSettings(**overrides)


def test_tracking_statistics_includes_tool_and_gripper_errors():
    samples = [
        _sample(
            0.0,
            np.array([0.003, 0.004, 0.0]),
            0.01,
            gripper_positions_rad=(np.array([0.2]), np.array([0.1])),
        ),
        _sample(
            1.0,
            np.array([0.0, 0.0, 0.010]),
            0.02,
            gripper_positions_rad=(np.array([0.4]), np.array([0.2])),
        ),
    ]

    stats = tracking_statistics(samples)

    assert stats.osc_position_m.rms == pytest.approx(np.sqrt((0.005**2 + 0.010**2) / 2.0))
    assert stats.osc_position_m.maximum == pytest.approx(0.010)
    assert stats.osc_orientation_rad.maximum == pytest.approx(0.02)
    assert stats.end_to_end_position_m.maximum == pytest.approx(0.020)
    assert stats.end_to_end_orientation_rad.maximum == pytest.approx(0.04)
    assert stats.gripper_position_rad[0].rms == pytest.approx(np.sqrt(0.025))


def test_command_acceleration_is_derived_from_position_setpoints():
    sample_rate_hz = 50.0
    acceleration_rad_s2 = 1.2
    native_samples = []
    for elapsed_s in np.arange(0.0, 4.0 + 1.0 / sample_rate_hz, 1.0 / sample_rate_hz):
        position = np.array([0.5 * acceleration_rad_s2 * elapsed_s**2])
        for stream, received_offset_s in (("controller", 0.001), ("state", 0.002)):
            native_samples.append(
                NativeJointSample(
                    segment="figure_eight",
                    setting="xz_1x",
                    window_index=0,
                    stream=stream,
                    received_timestamp_s=elapsed_s + received_offset_s,
                    source_timestamp_s=elapsed_s,
                    joint_names=("arm_1",),
                    joint_positions_rad=position,
                    # Deliberately inconsistent: this is a solver velocity/speed hint,
                    # not the derivative of the published position setpoint.
                    joint_velocities_rad_s=np.zeros(1),
                    tool_position_m=np.zeros(3),
                )
            )

    analysis = analyze_smoothness(
        native_samples,
        motion_segments=("figure_eight",),
        settings=ControllerTrackingSettings(derivative_smoothing_s=0.1),
        acceleration_limits_rad_s2={"arm_1": 1.0},
    )

    assert analysis is not None
    stats = analysis.statistics[0]
    assert stats.command_acceleration_rms_rad_s2 == pytest.approx(
        acceleration_rad_s2,
        rel=0.05,
    )
    assert stats.command_acceleration_p95_rad_s2 == pytest.approx(
        acceleration_rad_s2,
        rel=0.05,
    )
    assert stats.acceleration_limit_fraction is not None
    assert stats.acceleration_limit_fraction > MINIMUM_LIMIT_OCCUPANCY
    np.testing.assert_allclose(analysis.command_traces[0].raw_velocities_rad_s, 0.0)


def test_publication_statistics_and_csv_use_observed_intervals(tmp_path):
    timings = [
        ControllerCommandTiming(
            segment="figure_eight",
            setting="xz_1x",
            stream="controller",
            timestamp_s=10.0 + offset,
            source_timestamp_s=20.0 + offset,
        )
        for offset in (0.0, 0.03, 0.07, 0.10)
    ]

    stats = controller_publication_statistics(timings, target_rate_hz=30.0)
    assert stats.command_count == EXPECTED_TIMING_COMMAND_COUNT
    assert stats.mean_rate_hz == pytest.approx(30.0)
    assert stats.maximum_period_s == pytest.approx(0.04)

    path = tmp_path / "timing.csv"
    write_controller_timing_csv(path, timings, target_rate_hz=30.0)
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == EXPECTED_TIMING_COMMAND_COUNT
    assert float(rows[-1]["instantaneous_rate_hz"]) == pytest.approx(1.0 / 0.03)


def test_publication_statistics_exclude_gaps_between_settings():
    timings = [
        ControllerCommandTiming(
            segment="figure_eight",
            setting=setting,
            window_index=window_index,
            stream="robot",
            timestamp_s=timestamp,
        )
        for setting, window_index, timestamp in (
            ("xy_1x", 0, 10.00),
            ("xy_1x", 0, 10.04),
            ("xz_1x", 1, 20.00),
            ("xz_1x", 1, 20.04),
        )
    ]

    stats = controller_publication_statistics(timings, target_rate_hz=30.0)

    assert stats.mean_rate_hz == pytest.approx(25.0)
    assert stats.maximum_period_s == pytest.approx(0.04)


def test_csv_and_per_setting_svg_report_are_written(tmp_path):
    feedforward = CartesianVelocity(
        linear=np.array([0.01, -0.02, 0.03]),
        angular=np.array([0.1, -0.2, 0.3]),
    )
    samples = [
        _sample(
            0.0,
            np.array([0.002, 0.0, 0.0]),
            0.006,
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
            commanded_velocity=feedforward,
        ),
        _sample(
            1.0,
            np.array([0.001, 0.0, 0.0]),
            0.004,
            arm_joint_positions_rad=(np.array([0.3, -0.2]), np.array([0.28, -0.18])),
            gripper_positions_rad=(np.array([0.4]), np.array([0.35])),
            commanded_velocity=feedforward,
        ),
        _sample(
            2.0,
            np.array([0.0004, 0.0, 0.0]),
            0.002,
            segment="figure_eight_settle",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
            commanded_velocity=CartesianVelocity.zero(),
        ),
    ]
    csv_path = tmp_path / "tracking.csv"
    metrics_path = tmp_path / "metrics.json"

    write_tracking_csv(csv_path, samples)
    write_run_metrics_json(metrics_path, samples, analyses={})
    plot_paths = write_tracking_plots(
        tmp_path,
        samples,
        ControllerTrackingSettings(),
        "triskel",
        effort_traces={
            "xz_1x": ActuatorEffortTrace(
                joint_names=("arm_1", "arm_2", "gripper_1"),
                times_s=np.array([0.0, 1.0, 2.0]),
                efforts=np.array([[0.1, 0.2, 0.3], [0.9, 0.1, 0.2], [0.1, 0.1, 0.1]]),
                units=("N·m", "N·m", "N·m"),
                source="Current-based torque estimate",
            )
        },
        effort_limits={
            name: ActuatorEffortLimits(stall=2.942, rated=0.981)
            for name in ("arm_1", "arm_2", "gripper_1")
        },
    )

    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert [row["segment"] for row in rows] == [
        "figure_eight",
        "figure_eight",
        "figure_eight_settle",
    ]
    assert float(rows[1]["arm_1_controller_command_rad"]) == pytest.approx(0.3)
    assert float(rows[1]["gripper_1_error_rad"]) == pytest.approx(0.05)
    assert float(rows[0]["feedforward_wz_rad_s"]) == pytest.approx(0.3)
    assert float(rows[0]["osc_position_error_m"]) == pytest.approx(0.002)
    assert float(rows[0]["end_to_end_position_error_m"]) == pytest.approx(0.004)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    overall = metrics["tracking"]["overall"]
    assert overall["osc_tool_translation_m"]["maximum"] == pytest.approx(0.002)
    assert overall["end_to_end_tool_translation_m"]["maximum"] == pytest.approx(0.004)
    assert metrics["tracking"]["xz_1x"] == overall

    assert len(plot_paths) == 1
    assert plot_paths[0].name == "figure_eight_xz_1x.svg"
    svg = plot_paths[0].read_text(encoding="utf-8")
    assert "Controller tracking measurement" in svg
    assert "Cartesian figure eight — XZ plane, 1x size" in svg
    assert "OSC command FK vs reference" in svg
    assert "Measured FK vs OSC command FK" in svg
    assert "Measured FK vs reference (end-to-end)" in svg
    assert "Cartesian tracking error summary" in svg
    assert "Joint tracking error" in svg
    assert "Actuator effort" in svg
    assert "Current-based torque estimate" in svg
    assert "Maximum (stall): 2.94 N·m" in svg
    assert "Rated: 0.981 N·m" in svg
    assert "Cartesian velocity feedforward" in svg
    assert svg.index("Cartesian path comparisons") < svg.index("Joint tracking error")


def test_run_comparison_uses_previous_metrics_when_directory_is_baseline(tmp_path):
    baseline_path = tmp_path / "older" / "metrics.json"
    current_path = tmp_path / "current" / "metrics.json"
    baseline_path.parent.mkdir()
    current_path.parent.mkdir()
    baseline_path.write_text(
        json.dumps({"smoothness": {"figure_eight": {"jerk_rms_rad_s3": 10.0}}}),
        encoding="utf-8",
    )
    current_path.write_text(
        json.dumps({"smoothness": {"figure_eight": {"jerk_rms_rad_s3": 8.0}}}),
        encoding="utf-8",
    )
    comparison_path = tmp_path / "comparison.md"

    selected_baseline = write_run_comparison(comparison_path, current_path, tmp_path)

    assert selected_baseline == baseline_path
    assert "-20.0%" in comparison_path.read_text(encoding="utf-8")
