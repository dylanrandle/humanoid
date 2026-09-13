import csv
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pinocchio as pin
import pytest

import humanoid.robots.utils.controller_tracking.cli as cli_module
import humanoid.robots.utils.controller_tracking.report as report_module
import humanoid.robots.utils.controller_tracking.runtime as runtime_module
import humanoid.robots.utils.controller_tracking.sampling as sampling_module
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.controllers.operational_space import OperationalSpaceController
from humanoid.robots.base import Robot
from humanoid.robots.utils.controller_tracking import (
    DEFAULT_COMMAND_RATE_HZ,
    DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION,
    DEFAULT_JOINT_CYCLES,
    ControllerCommandTiming,
    ControllerTrackingSettings,
    Segment,
    TrackingRun,
    TrackingSample,
    controller_publication_statistics,
    figure_eight_offset,
    gripper_sinusoid,
    interpolated_cartesian_comparison_pose,
    joint_space_targets,
    resolve_tracking_comparison,
    tracking_statistics,
    write_controller_timing_csv,
    write_tracking_csv,
    write_tracking_plots,
)
from humanoid.robots.utils.controller_tracking.endpoints import (
    TRISKEL_CONTROLLER_TRACKING_COMPARISON,
)
from humanoid.robots.utils.controller_tracking.models import RuntimeFeedback
from humanoid.robots.utils.controller_tracking.timing import ControllerCommandTimingRecorder
from humanoid.types.homing import HomingPreset
from humanoid.types.orchestrator import Mode
from humanoid.types.robot import RobotJointCommand, RobotState

EXPECTED_JOINT_PLOTS_PER_SECTION = 3
EXPECTED_COMPARISON_SUBSECTIONS = 2
EXPECTED_CSV_REPORTS = 2
EXPECTED_REPORTS = 3
EXPECTED_TIMING_COMMAND_COUNT = 4


def _sample(  # noqa: PLR0913 - test builder exposes independent sample dimensions
    elapsed_s: float,
    position_error_m: np.ndarray,
    orientation_error_rad: float,
    *,
    segment: Segment = "joint_comparison",
    arm_joint_positions_rad: tuple[np.ndarray, np.ndarray] | None = None,
    gripper_positions_rad: tuple[np.ndarray, np.ndarray] | None = None,
) -> TrackingSample:
    commanded = np.array([0.1 + elapsed_s, 0.2, 0.3])
    commanded_gripper_positions_rad, measured_gripper_positions_rad = (
        gripper_positions_rad if gripper_positions_rad is not None else (None, None)
    )
    gripper_error = (
        commanded_gripper_positions_rad - measured_gripper_positions_rad
        if commanded_gripper_positions_rad is not None
        and measured_gripper_positions_rad is not None
        else None
    )
    gripper_joint_names = (
        tuple(f"test_gripper_{index + 1}" for index in range(len(commanded_gripper_positions_rad)))
        if commanded_gripper_positions_rad is not None
        else ()
    )
    commanded_arm_joint_positions_rad, measured_arm_joint_positions_rad = (
        arm_joint_positions_rad
        if arm_joint_positions_rad is not None
        else (np.empty(0), np.empty(0))
    )
    arm_joint_position_errors_rad = np.arctan2(
        np.sin(commanded_arm_joint_positions_rad - measured_arm_joint_positions_rad),
        np.cos(commanded_arm_joint_positions_rad - measured_arm_joint_positions_rad),
    )
    return TrackingSample(
        segment=segment,
        elapsed_s=elapsed_s,
        state_timestamp_s=100.0 + elapsed_s,
        commanded_position_m=commanded,
        measured_position_m=commanded - position_error_m,
        position_error_m=position_error_m,
        orientation_error_rad=orientation_error_rad,
        joint_command_timestamp_s=99.98 + elapsed_s,
        state_minus_joint_command_s=0.02,
        arm_joint_names=("arm_1", "arm_2") if arm_joint_positions_rad is not None else (),
        commanded_arm_joint_positions_rad=commanded_arm_joint_positions_rad,
        measured_arm_joint_positions_rad=measured_arm_joint_positions_rad,
        arm_joint_position_errors_rad=arm_joint_position_errors_rad,
        commanded_arm_joint_velocities_rad_s=None,
        measured_arm_joint_velocities_rad_s=np.zeros_like(measured_arm_joint_positions_rad),
        gripper_joint_names=gripper_joint_names,
        commanded_gripper_positions_rad=commanded_gripper_positions_rad,
        measured_gripper_positions_rad=measured_gripper_positions_rad,
        gripper_position_errors_rad=gripper_error,
        commanded_gripper_velocities_rad_s=None,
        measured_gripper_velocities_rad_s=(
            np.zeros_like(measured_gripper_positions_rad)
            if measured_gripper_positions_rad is not None
            else None
        ),
    )


def test_default_command_rate_matches_controller_rate():
    expected_rate_hz = 30.0
    settings = ControllerTrackingSettings()

    assert settings.rate_hz == DEFAULT_COMMAND_RATE_HZ == expected_rate_hz
    assert settings.comparison_duration_s == pytest.approx(8.0)
    assert settings.joint_cycles == DEFAULT_JOINT_CYCLES
    assert settings.move_gripper
    assert settings.gripper_limit_margin_fraction == DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION


def test_controller_publication_statistics_report_rate_jitter_and_delays():
    timings = [
        ControllerCommandTiming(segment="figure_eight", timestamp_s=timestamp)
        for timestamp in (10.0, 10.03, 10.06, 10.12)
    ]

    stats = controller_publication_statistics(timings, target_rate_hz=30.0)

    assert stats.command_count == EXPECTED_TIMING_COMMAND_COUNT
    assert stats.mean_rate_hz == pytest.approx(25.0)
    assert stats.median_period_s == pytest.approx(0.03)
    assert stats.maximum_period_s == pytest.approx(0.06)
    assert stats.period_jitter_s > 0.0
    assert stats.delayed_interval_count == 1


def test_command_to_state_timing_ignores_samples_with_stale_command_timestamp():
    first = replace(
        _sample(0.0, np.zeros(3), 0.0),
        joint_command_timestamp_s=10.0,
        state_minus_joint_command_s=0.02,
    )
    stale = replace(
        _sample(1.0, np.zeros(3), 0.0),
        joint_command_timestamp_s=10.0,
        state_minus_joint_command_s=1.02,
    )
    fresh = replace(
        _sample(2.0, np.zeros(3), 0.0),
        joint_command_timestamp_s=12.0,
        state_minus_joint_command_s=0.03,
    )

    offsets = cli_module._fresh_command_to_state_offsets_s([first, stale, fresh])

    np.testing.assert_allclose(offsets, [0.02, 0.03])


def test_joint_plot_measurements_use_the_reported_error_branch():
    sample = _sample(
        0.0,
        np.zeros(3),
        0.0,
        arm_joint_positions_rad=(
            np.array([3.1, 0.5]),
            np.array([-3.1, 0.45]),
        ),
        gripper_positions_rad=(np.array([0.2]), np.array([0.18])),
    )

    commands, aligned_measurements, names = report_module._aligned_joint_position_matrices([sample])

    assert names == ("arm_1", "arm_2", "test_gripper_1")
    assert sample.gripper_position_errors_rad is not None
    np.testing.assert_allclose(
        commands - aligned_measurements,
        np.hstack((sample.arm_joint_position_errors_rad, sample.gripper_position_errors_rad))[
            None, :
        ],
    )
    assert aligned_measurements[0, 0] == pytest.approx(-3.1 + 2.0 * np.pi)


def test_controller_timing_recorder_keeps_every_publication_inside_marked_window():
    subscriber = MagicMock()
    subscriber.receive.side_effect = [
        RobotJointCommand(timestamp=0.9, joint_positions=np.empty(0)),
        RobotJointCommand(timestamp=1.0, joint_positions=np.empty(0)),
        RobotJointCommand(timestamp=1.03, joint_positions=np.empty(0)),
        RobotJointCommand(timestamp=1.06, joint_positions=np.empty(0)),
        RobotJointCommand(timestamp=1.2, joint_positions=np.empty(0)),
        None,
    ]
    clock = iter((1.0, 1.1))
    recorder = ControllerCommandTimingRecorder(
        subscriber=subscriber,
        clock=lambda: next(clock),
    )

    recorder.begin("figure_eight")
    recorder.end()
    timings = recorder.close()

    assert [timing.timestamp_s for timing in timings] == [1.0, 1.03, 1.06]
    subscriber.close.assert_called_once()


def test_controller_timing_csv_preserves_publication_intervals(tmp_path):
    timings = [
        ControllerCommandTiming(segment="figure_eight", timestamp_s=10.0),
        ControllerCommandTiming(segment="figure_eight", timestamp_s=10.03),
        ControllerCommandTiming(segment="figure_eight", timestamp_s=10.09),
    ]
    timing_path = tmp_path / "timing.csv"

    write_controller_timing_csv(timing_path, timings, target_rate_hz=30.0)

    with timing_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert [float(row["segment_elapsed_s"]) for row in rows] == pytest.approx([0.0, 0.03, 0.09])
    assert rows[0]["period_s"] == ""
    assert float(rows[1]["instantaneous_rate_hz"]) == pytest.approx(1.0 / 0.03)
    assert rows[1]["delayed_interval"] == "False"
    assert rows[2]["delayed_interval"] == "True"


def test_figure_eight_uses_selected_plane_and_dimensions():
    settings = ControllerTrackingSettings(
        plane="xz",
        width_m=0.08,
        height_m=0.04,
        period_s=8.0,
        cycles=1,
        ramp_s=0.0,
    )

    quarter_cycle = figure_eight_offset(2.0, settings)
    eighth_cycle = figure_eight_offset(1.0, settings)

    np.testing.assert_allclose(quarter_cycle, [0.04, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        eighth_cycle,
        [0.04 / np.sqrt(2.0), 0.0, 0.02],
        atol=1e-12,
    )


def test_figure_eight_ramp_returns_to_anchor():
    settings = ControllerTrackingSettings(period_s=4.0, cycles=2, ramp_s=1.0)

    np.testing.assert_allclose(figure_eight_offset(0.0, settings), np.zeros(3))
    np.testing.assert_allclose(
        figure_eight_offset(settings.duration_s, settings),
        np.zeros(3),
        atol=1e-12,
    )
    assert np.linalg.norm(figure_eight_offset(0.5, settings)) > 0.0


def test_resolved_comparison_uses_supplied_corresponding_endpoints():
    robot = Robot(TRISKEL_CONFIG)
    resolved = resolve_tracking_comparison(robot, TRISKEL_CONTROLLER_TRACKING_COMPARISON)

    for endpoint, joint_positions, task_pose in (
        (
            TRISKEL_CONTROLLER_TRACKING_COMPARISON.start,
            resolved.start_joint_positions,
            resolved.start_task_pose,
        ),
        (
            TRISKEL_CONTROLLER_TRACKING_COMPARISON.end,
            resolved.end_joint_positions,
            resolved.end_task_pose,
        ),
    ):
        for joint_name, expected_position in endpoint.joint_positions_rad.items():
            measured_position = robot.joint_position_from_q(
                joint_positions,
                robot.joint_name_to_idx(joint_name),
            )
            assert measured_position == pytest.approx(expected_position)
        assert robot.get_tool_command_pose(joint_positions).isApprox(task_pose, 1e-12)


def test_cartesian_comparison_interpolation_reaches_endpoints_and_stops_smoothly():
    robot = Robot(TRISKEL_CONFIG)
    comparison = resolve_tracking_comparison(robot, TRISKEL_CONTROLLER_TRACKING_COMPARISON)
    duration_s = 4.0
    epsilon_s = 1e-3

    at_start = interpolated_cartesian_comparison_pose(
        comparison.start_task_pose,
        comparison.end_task_pose,
        0.0,
        duration_s,
    )
    after_start = interpolated_cartesian_comparison_pose(
        comparison.start_task_pose,
        comparison.end_task_pose,
        epsilon_s,
        duration_s,
    )
    before_end = interpolated_cartesian_comparison_pose(
        comparison.start_task_pose,
        comparison.end_task_pose,
        duration_s - epsilon_s,
        duration_s,
    )
    at_end = interpolated_cartesian_comparison_pose(
        comparison.start_task_pose,
        comparison.end_task_pose,
        duration_s,
        duration_s,
    )
    at_midpoint = interpolated_cartesian_comparison_pose(
        comparison.start_task_pose,
        comparison.end_task_pose,
        duration_s / 2.0,
        duration_s,
    )

    assert at_start.isApprox(comparison.start_task_pose, 1e-12)
    assert at_end.isApprox(comparison.end_task_pose, 1e-12)
    np.testing.assert_allclose(
        at_midpoint.translation,
        (comparison.start_task_pose.translation + comparison.end_task_pose.translation) / 2.0,
    )
    start_speed = np.linalg.norm(pin.log6(at_start.inverse() * after_start).vector) / epsilon_s
    end_speed = np.linalg.norm(pin.log6(before_end.inverse() * at_end).vector) / epsilon_s
    maximum_boundary_speed = 1e-3
    assert start_speed < maximum_boundary_speed
    assert end_speed < maximum_boundary_speed


def test_gripper_sinusoid_cycles_between_bounds_and_returns_to_initial_position():
    settings = ControllerTrackingSettings(
        period_s=4.0,
        cycles=1,
        ramp_s=1.0,
        gripper_period_s=4.0,
    )
    initial = np.array([0.3])
    lower = np.array([0.1])
    upper = np.array([0.9])

    np.testing.assert_allclose(gripper_sinusoid(0.0, initial, lower, upper, settings), initial)
    np.testing.assert_allclose(gripper_sinusoid(1.0, initial, lower, upper, settings), upper)
    np.testing.assert_allclose(gripper_sinusoid(3.0, initial, lower, upper, settings), lower)
    np.testing.assert_allclose(
        gripper_sinusoid(settings.duration_s, initial, lower, upper, settings), initial
    )


def test_joint_space_targets_start_and_end_home_with_requested_round_trips():
    targets = joint_space_targets(TRISKEL_CONFIG, cycles=2)

    assert [preset for preset, _ in targets] == [
        HomingPreset.HOME,
        HomingPreset.REST,
        HomingPreset.HOME,
        HomingPreset.REST,
        HomingPreset.HOME,
    ]
    np.testing.assert_array_equal(targets[1][1], TRISKEL_CONFIG.homing_presets[HomingPreset.REST])
    np.testing.assert_array_equal(targets[-1][1], TRISKEL_CONFIG.homing_presets[HomingPreset.HOME])
    assert joint_space_targets(TRISKEL_CONFIG, cycles=0) == ()


def test_home_convergence_waits_for_stable_encoder_feedback(monkeypatch):
    robot = Robot(TRISKEL_CONFIG)
    home = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME]
    arm_joint_indices = robot.get_arm_joint_indices()
    converged_timestamp = 4.0

    def offset_arm_positions(offset_rad: float) -> np.ndarray:
        positions = {}
        for joint_index in range(robot.model.njoints - 1):
            try:
                position = robot.joint_position_from_q(home, joint_index)
            except ValueError:
                continue
            positions[joint_index] = position + (
                offset_rad if joint_index in arm_joint_indices else 0.0
            )
        return robot.joint_positions_to_q(positions)

    def feedback(timestamp: float, arm_offset_rad: float) -> RuntimeFeedback:
        received_at = timestamp
        return RuntimeFeedback(
            state=RobotState(
                timestamp=timestamp,
                joint_positions=offset_arm_positions(arm_offset_rad),
                joint_velocities=np.zeros(robot.model.nv),
                actuator_temperatures=np.empty(0),
            ),
            joint_command=RobotJointCommand(timestamp=0.0, joint_positions=home.copy()),
            mode=Mode.IDLE,
            last_state_received_s=received_at,
            last_joint_command_received_s=received_at,
            last_mode_received_s=received_at,
        )

    initial_feedback = feedback(1.0, 0.1)
    updates = iter(
        [
            feedback(2.0, 0.1),
            feedback(3.0, 0.01),
            feedback(converged_timestamp, 0.01),
        ]
    )
    clock = iter(np.arange(0.0, 10.0, 0.05).tolist())
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        runtime_module,
        "_refresh_runtime_state",
        lambda *_args, **_kwargs: next(updates),
    )
    monkeypatch.setattr(
        runtime_module,
        "_sleep_until_next_tick",
        lambda previous_tick, _rate_hz: previous_tick,
    )
    samples: list[TrackingSample] = []

    converged = runtime_module._wait_for_home_convergence(
        ControllerTrackingSettings(
            home_position_tolerance_rad=0.03,
            home_stable_s=0.1,
            home_timeout_s=2.0,
        ),
        robot,
        None,  # type: ignore[arg-type] - feedback refresh is replaced above
        initial_feedback,
        home,
        samples,
        run_started_s=0.0,
    )

    assert converged.state.timestamp == converged_timestamp
    assert [sample.state_timestamp_s for sample in samples] == [2.0, 3.0, converged_timestamp]


def test_comparison_joint_motion_uses_homing_controller(monkeypatch):
    robot = Robot(TRISKEL_CONFIG)
    home = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME]
    comparison = resolve_tracking_comparison(robot, TRISKEL_CONTROLLER_TRACKING_COMPARISON)
    target = comparison.end_joint_positions
    feedback = RuntimeFeedback(
        state=RobotState(
            timestamp=1.0,
            joint_positions=home.copy(),
            joint_velocities=np.zeros(robot.model.nv),
            actuator_temperatures=np.empty(0),
        ),
        joint_command=RobotJointCommand(timestamp=1.0, joint_positions=home.copy()),
        mode=Mode.IDLE,
        last_state_received_s=1.0,
        last_joint_command_received_s=1.0,
        last_mode_received_s=1.0,
    )
    orchestrator = MagicMock()
    monkeypatch.setattr(runtime_module, "_wait_for_mode", lambda *_args: None)
    monkeypatch.setattr(
        runtime_module,
        "_wait_for_fresh_joint_command",
        lambda *_args, **_kwargs: RobotJointCommand(
            timestamp=2.0,
            joint_positions=target.copy(),
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "_sleep_until_next_tick",
        lambda previous_tick, _rate_hz: previous_tick,
    )
    monkeypatch.setattr(
        runtime_module,
        "_refresh_runtime_state",
        lambda _subscriber, current, *_args, **_kwargs: runtime_module.replace(
            current,
            mode=Mode.IDLE,
        ),
    )
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: 10.0)
    samples: list[TrackingSample] = []

    runtime_module._run_homing_target(
        "comparison end",
        target,
        "joint_comparison",
        ControllerTrackingSettings(),
        robot,
        orchestrator,
        None,  # type: ignore[arg-type] - runtime refresh is replaced above
        feedback,
        samples,
        run_started_s=0.0,
    )

    orchestrator.request_homing.assert_called_once_with(target)
    assert [sample.segment for sample in samples] == ["joint_comparison"]
    expected_arm_positions = np.array(
        [
            robot.joint_position_from_q(target, joint_index)
            for joint_index in robot.get_arm_joint_indices()
        ]
    )
    np.testing.assert_allclose(samples[0].commanded_arm_joint_positions_rad, expected_arm_positions)


def test_tracking_sample_captures_commanded_and_measured_joint_velocities():
    robot = Robot(TRISKEL_CONFIG)
    positions = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME]
    arm_joint_indices = robot.get_arm_joint_indices()
    gripper_joint_indices = robot.get_gripper_joint_indices()
    measured_velocity_by_joint = {
        joint_index: 0.1 * (offset + 1)
        for offset, joint_index in enumerate((*arm_joint_indices, *gripper_joint_indices))
    }
    measured_velocities = robot.joint_velocities_to_v(measured_velocity_by_joint)
    commanded_velocities = measured_velocities * 2.0
    gripper_positions = positions[robot.get_gripper_position_indices()].copy()
    feedback = RuntimeFeedback(
        state=RobotState(
            timestamp=2.0,
            joint_positions=positions.copy(),
            joint_velocities=measured_velocities,
            actuator_temperatures=np.empty(0),
        ),
        joint_command=RobotJointCommand(
            timestamp=1.98,
            joint_positions=positions.copy(),
            joint_velocities=commanded_velocities,
        ),
        mode=Mode.SYSTEM,
        last_state_received_s=2.0,
        last_joint_command_received_s=2.0,
        last_mode_received_s=2.0,
    )

    sample = sampling_module._tracking_sample(
        "figure_eight",
        1.0,
        robot.get_tool_command_pose(positions),
        gripper_positions,
        feedback,
        robot,
    )

    arm_velocity_indices = robot.get_joint_velocity_indices(arm_joint_indices)
    gripper_velocity_indices = robot.get_joint_velocity_indices(gripper_joint_indices)
    np.testing.assert_array_equal(
        sample.commanded_arm_joint_velocities_rad_s,
        commanded_velocities[arm_velocity_indices],
    )
    np.testing.assert_array_equal(
        sample.measured_arm_joint_velocities_rad_s,
        measured_velocities[arm_velocity_indices],
    )
    np.testing.assert_array_equal(
        sample.commanded_gripper_velocities_rad_s,
        commanded_velocities[gripper_velocity_indices],
    )
    np.testing.assert_array_equal(
        sample.measured_gripper_velocities_rad_s,
        measured_velocities[gripper_velocity_indices],
    )


def test_triskel_tuning_tracks_default_figure_eight_with_submillimeter_rms():
    robot = Robot(TRISKEL_CONFIG)
    config = TRISKEL_CONFIG.operational_space_config
    assert config is not None
    controller = OperationalSpaceController(robot, config)
    q = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy()
    controller.update_state(q)
    anchor = robot.get_tool_command_pose(q)
    settings = ControllerTrackingSettings(cycles=1, start_delay_s=0.0, settle_s=0.0)
    errors = []

    for step in range(round(settings.duration_s * settings.rate_hz) + 1):
        elapsed_s = step / settings.rate_hz
        target = pin.SE3(
            anchor.rotation.copy(),
            anchor.translation + figure_eight_offset(elapsed_s, settings),
        )
        result = controller.compute_control(target, dt=1.0 / settings.rate_hz)
        actual = robot.get_tool_command_pose(result.q)
        errors.append(np.linalg.norm(target.translation - actual.translation))

    rms_error_m = float(np.sqrt(np.mean(np.square(errors))))
    maximum_rms_error_m = 1e-3
    assert rms_error_m < maximum_rms_error_m


def test_triskel_tuning_tracks_supplied_cartesian_comparison_with_low_rms():
    robot = Robot(TRISKEL_CONFIG)
    config = TRISKEL_CONFIG.operational_space_config
    assert config is not None
    controller = OperationalSpaceController(robot, config)
    comparison = resolve_tracking_comparison(robot, TRISKEL_CONTROLLER_TRACKING_COMPARISON)
    controller.update_state(comparison.start_joint_positions)
    settings = ControllerTrackingSettings(cycles=1, start_delay_s=0.0, settle_s=0.0)
    errors = []

    for step in range(round(settings.comparison_duration_s * settings.rate_hz) + 1):
        elapsed_s = step / settings.rate_hz
        target = interpolated_cartesian_comparison_pose(
            comparison.start_task_pose,
            comparison.end_task_pose,
            elapsed_s,
            settings.comparison_duration_s,
        )
        result = controller.compute_control(target, dt=1.0 / settings.rate_hz)
        actual = robot.get_tool_command_pose(result.q)
        errors.append(np.linalg.norm(target.translation - actual.translation))

    rms_error_m = float(np.sqrt(np.mean(np.square(errors))))
    maximum_rms_error_m = 2e-3
    assert rms_error_m < maximum_rms_error_m


@pytest.mark.parametrize(
    "settings",
    [
        {"width_m": 0.0},
        {"height_m": 0.0},
        {"cycles": 0},
        {"comparison_duration_s": 0.0},
        {"joint_cycles": -1},
        {"ramp_s": -1.0},
        {"period_s": 2.0, "cycles": 1, "ramp_s": 1.1},
        {"home_position_tolerance_rad": 0.0},
        {"home_stable_s": 0.0},
        {"home_timeout_s": 0.0},
        {"gripper_min_rad": 0.1},
        {"gripper_min_rad": 0.2, "gripper_max_rad": 0.1},
        {"gripper_period_s": 0.0},
        {"gripper_limit_margin_fraction": 0.5},
        {"move_gripper": False, "gripper_period_s": 1.0},
    ],
)
def test_settings_reject_invalid_motion_parameters(settings):
    with pytest.raises(ValueError):
        ControllerTrackingSettings(**settings)


def test_tracking_statistics_reports_norms_and_percentiles():
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

    assert stats.position_m.rms == pytest.approx(np.sqrt((0.005**2 + 0.010**2) / 2.0))
    assert stats.position_m.mean == pytest.approx(0.0075)
    assert stats.position_m.p95 == pytest.approx(0.00975)
    assert stats.position_m.maximum == pytest.approx(0.010)
    assert stats.orientation_rad.maximum == pytest.approx(0.02)
    assert stats.arm_joint_position_rad == ()
    assert stats.gripper_position_rad[0].rms == pytest.approx(np.sqrt(0.025))
    assert stats.gripper_position_rad[0].maximum == pytest.approx(0.2)


def test_tracking_statistics_rejects_empty_samples():
    with pytest.raises(ValueError, match="empty"):
        tracking_statistics([])


def test_csv_and_separate_svg_reports_are_written(  # noqa: PLR0915 - verifies all reports
    tmp_path,
):
    samples = [
        _sample(
            0.0,
            np.array([0.001, 0.0, 0.0]),
            0.006,
            segment="joint_home",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
        ),
        _sample(
            1.0,
            np.array([0.001, 0.0, 0.0]),
            0.006,
            segment="joint_rest",
            arm_joint_positions_rad=(np.array([-0.2, 0.3]), np.array([-0.18, 0.27])),
            gripper_positions_rad=(np.array([0.1]), np.array([0.12])),
        ),
        _sample(
            2.0,
            np.array([0.001, 0.0, 0.0]),
            0.006,
            segment="joint_home",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
        ),
        _sample(
            3.0,
            np.array([0.001, 0.0, 0.0]),
            0.01,
            segment="figure_eight",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.08])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.18])),
        ),
        _sample(
            4.0,
            np.array([0.002, 0.0, 0.0]),
            0.02,
            segment="figure_eight",
            arm_joint_positions_rad=(np.array([0.4, -0.2]), np.array([0.35, -0.18])),
            gripper_positions_rad=(np.array([0.4]), np.array([0.35])),
        ),
        _sample(
            5.0,
            np.array([0.0005, 0.0, 0.0]),
            0.005,
            segment="figure_eight_settle",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
        ),
        _sample(
            6.0,
            np.array([0.0015, 0.0, 0.0]),
            0.008,
            segment="joint_comparison",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.18, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.17])),
        ),
        _sample(
            7.0,
            np.array([0.0004, 0.0, 0.0]),
            0.004,
            segment="joint_comparison_settle",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
        ),
        _sample(
            8.0,
            np.array([0.0015, 0.0, 0.0]),
            0.008,
            segment="cartesian_comparison",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.18, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.17])),
        ),
        _sample(
            9.0,
            np.array([0.0004, 0.0, 0.0]),
            0.004,
            segment="cartesian_comparison_settle",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
        ),
    ]
    settings = ControllerTrackingSettings(period_s=1.0, cycles=1, ramp_s=0.0, settle_s=1.0)
    csv_path = tmp_path / "tracking.csv"

    write_tracking_csv(csv_path, samples)
    plot_paths = write_tracking_plots(tmp_path, "tracking", samples, settings, "triskel")

    with csv_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == len(samples)
    assert rows[0]["segment"] == "joint_home"
    assert rows[1]["segment"] == "joint_rest"
    assert rows[2]["segment"] == "joint_home"
    assert rows[5]["segment"] == "figure_eight_settle"
    assert rows[6]["segment"] == "joint_comparison"
    assert rows[7]["segment"] == "joint_comparison_settle"
    assert rows[8]["segment"] == "cartesian_comparison"
    assert rows[-1]["segment"] == "cartesian_comparison_settle"
    assert float(rows[4]["joint_command_timestamp_s"]) == pytest.approx(103.98)
    assert float(rows[1]["state_minus_joint_command_s"]) == pytest.approx(0.02)
    assert float(rows[4]["arm_1_controller_command_rad"]) == pytest.approx(0.4)
    assert float(rows[4]["arm_1_measured_rad"]) == pytest.approx(0.35)
    assert float(rows[4]["arm_1_error_rad"]) == pytest.approx(0.05)
    assert float(rows[4]["arm_1_controller_command_rad_s"]) == pytest.approx(0.2)
    assert float(rows[4]["arm_1_measured_rad_s"]) == pytest.approx(0.0)
    assert float(rows[4]["position_error_m"]) == pytest.approx(0.002)
    assert float(rows[4]["test_gripper_1_error_rad"]) == pytest.approx(0.05)

    assert plot_paths.home_rest.name == "tracking_home_rest.svg"
    assert plot_paths.figure_eight.name == "tracking_figure_eight.svg"
    assert plot_paths.comparison.name == "tracking_comparison.svg"
    home_rest_svg = plot_paths.home_rest.read_text(encoding="utf-8")
    figure_eight_svg = plot_paths.figure_eight.read_text(encoding="utf-8")
    comparison_svg = plot_paths.comparison.read_text(encoding="utf-8")

    for svg_path in plot_paths:
        svg = svg_path.read_text(encoding="utf-8")
        assert "Controller tracking measurement" in svg
        assert "commanded" in svg
        assert "measured" in svg
        assert "arm_1 (rad)" in svg
        assert "joint position (rad)" in svg
        assert "joint velocity (rad/s)" in svg
        assert "Commanded vs measured joint velocities" in svg
        assert 'class="joint-error"' not in svg
        assert "test_gripper_1 (rad)" in svg
        assert "Gripper command and measurement" not in svg
        assert "RMS" in svg

    assert "Joint-space home/rest tracking" in home_rest_svg
    assert "Cartesian tracking error" not in home_rest_svg
    assert home_rest_svg.count("Joint tracking error") == 1
    assert home_rest_svg.count('class="joint-command"') == 2 * (
        EXPECTED_JOINT_PLOTS_PER_SECTION + 1
    )

    assert "Cartesian-space figure eight (OSC/IK)" in figure_eight_svg
    assert figure_eight_svg.count("Joint tracking error") == 1
    assert figure_eight_svg.count("Cartesian tracking error") == 1
    assert figure_eight_svg.count("Cartesian tracking plots") == 1
    assert figure_eight_svg.count('class="joint-command"') == 2 * (
        EXPECTED_JOINT_PLOTS_PER_SECTION + 1
    )

    direct_phase_title = "A · Joint-space trajectory (homing controller)"
    cartesian_phase_title = "B · Cartesian trajectory (OSC/IK)"
    assert "Same-endpoint command-space comparison" in comparison_svg
    assert direct_phase_title in comparison_svg
    assert cartesian_phase_title in comparison_svg
    assert comparison_svg.index(direct_phase_title) < comparison_svg.index(cartesian_phase_title)
    assert comparison_svg.count("Joint tracking error") == EXPECTED_COMPARISON_SUBSECTIONS
    assert comparison_svg.count("Cartesian tracking error") == EXPECTED_COMPARISON_SUBSECTIONS
    assert comparison_svg.count("Cartesian tracking plots") == EXPECTED_COMPARISON_SUBSECTIONS
    assert comparison_svg.count('class="joint-command"') == (
        2 * EXPECTED_COMPARISON_SUBSECTIONS * (EXPECTED_JOINT_PLOTS_PER_SECTION + 1)
    )
    assert comparison_svg.count('class="joint-measured"') == (
        2 * EXPECTED_COMPARISON_SUBSECTIONS * (EXPECTED_JOINT_PLOTS_PER_SECTION + 1)
    )


def test_partial_joint_space_report_keeps_unreached_comparison_sections_empty(tmp_path):
    samples = [
        _sample(
            0.0,
            np.array([0.001, 0.0, 0.0]),
            0.006,
            segment="joint_home",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
        ),
        _sample(
            1.0,
            np.array([0.001, 0.0, 0.0]),
            0.006,
            segment="joint_rest",
            arm_joint_positions_rad=(np.array([-0.2, 0.3]), np.array([-0.18, 0.27])),
            gripper_positions_rad=(np.array([0.1]), np.array([0.12])),
        ),
    ]
    plot_paths = write_tracking_plots(
        tmp_path,
        "partial_tracking",
        samples,
        ControllerTrackingSettings(),
        "triskel",
    )

    home_rest_svg = plot_paths.home_rest.read_text(encoding="utf-8")
    figure_eight_svg = plot_paths.figure_eight.read_text(encoding="utf-8")
    comparison_svg = plot_paths.comparison.read_text(encoding="utf-8")
    assert "Joint-space home/rest tracking" in home_rest_svg
    assert home_rest_svg.count("Joint tracking error") == 1
    assert "The Cartesian figure-eight phase was not reached" in figure_eight_svg
    assert "Cartesian tracking error" not in figure_eight_svg
    assert "The homing-controller comparison phase was not reached" in comparison_svg
    assert "The Cartesian OSC/IK comparison phase was not reached" in comparison_svg
    assert "Cartesian tracking plots" not in comparison_svg


def test_joint_plot_bounds_do_not_magnify_small_static_offsets():
    bounds = report_module._padded_bounds(
        np.array([0.0, 0.25]),
        minimum_span=report_module.MIN_JOINT_PLOT_SPAN_RAD,
    )

    assert bounds[1] - bounds[0] >= report_module.MIN_JOINT_PLOT_SPAN_RAD


def test_main_writes_partial_run_before_exiting_with_failure(monkeypatch, tmp_path):
    sample = _sample(
        0.0,
        np.array([0.001, 0.0, 0.0]),
        0.006,
        segment="joint_home",
        arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
        gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
    )
    monkeypatch.setattr(
        cli_module,
        "run_controller_tracking",
        lambda *_args, **_kwargs: TrackingRun(
            samples=[sample],
            completed=False,
            failure_reason="HOME convergence timed out",
        ),
    )
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        ["controller_tracking", "--output-dir", str(tmp_path), "--start-delay", "0"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()

    assert exc_info.value.code == 1
    csv_paths = list(tmp_path.glob("*.csv"))
    svg_paths = list(tmp_path.glob("*.svg"))
    assert len(csv_paths) == EXPECTED_CSV_REPORTS
    assert len(svg_paths) == EXPECTED_REPORTS
    for svg_path in svg_paths:
        svg = svg_path.read_text(encoding="utf-8")
        assert "Partial run" in svg
        assert "HOME convergence timed out" in svg
