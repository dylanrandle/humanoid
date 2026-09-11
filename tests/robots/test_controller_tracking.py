import csv

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.controllers.operational_space import OperationalSpaceController
from humanoid.robots.base import Robot
from humanoid.robots.utils.controller_tracking import (
    DEFAULT_COMMAND_RATE_HZ,
    DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION,
    ControllerTrackingSettings,
    Segment,
    TrackingSample,
    figure_eight_offset,
    gripper_sinusoid,
    tracking_statistics,
    write_tracking_csv,
    write_tracking_svg,
)
from humanoid.types.homing import HomingPreset

EXPECTED_ARM_JOINT_PLOTS = 2


def _sample(  # noqa: PLR0913 - test builder exposes independent sample dimensions
    elapsed_s: float,
    position_error_m: np.ndarray,
    orientation_error_rad: float,
    *,
    segment: Segment = "trajectory",
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
        commanded_gripper_positions_rad=commanded_gripper_positions_rad,
        measured_gripper_positions_rad=measured_gripper_positions_rad,
        gripper_position_errors_rad=gripper_error,
    )


def test_default_command_rate_matches_conservative_controller_rate():
    expected_rate_hz = 10.0
    settings = ControllerTrackingSettings()

    assert settings.rate_hz == DEFAULT_COMMAND_RATE_HZ == expected_rate_hz
    assert settings.move_gripper
    assert settings.gripper_limit_margin_fraction == DEFAULT_GRIPPER_LIMIT_MARGIN_FRACTION


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


def test_ramp_returns_trajectory_to_anchor_with_zero_endpoint_offset():
    settings = ControllerTrackingSettings(period_s=4.0, cycles=2, ramp_s=1.0)

    np.testing.assert_allclose(figure_eight_offset(0.0, settings), np.zeros(3))
    np.testing.assert_allclose(
        figure_eight_offset(settings.duration_s, settings), np.zeros(3), atol=1e-12
    )
    assert np.linalg.norm(figure_eight_offset(0.5, settings)) > 0.0


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


@pytest.mark.parametrize(
    "settings",
    [
        {"width_m": 0.0},
        {"cycles": 0},
        {"ramp_s": -1.0},
        {"period_s": 2.0, "cycles": 1, "ramp_s": 1.1},
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


def test_csv_and_svg_reports_are_written(tmp_path):
    samples = [
        _sample(
            0.0,
            np.array([0.001, 0.0, 0.0]),
            0.01,
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.08])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.18])),
        ),
        _sample(
            1.0,
            np.array([0.002, 0.0, 0.0]),
            0.02,
            arm_joint_positions_rad=(np.array([0.4, -0.2]), np.array([0.35, -0.18])),
            gripper_positions_rad=(np.array([0.4]), np.array([0.35])),
        ),
        _sample(
            2.0,
            np.array([0.0005, 0.0, 0.0]),
            0.005,
            segment="settle",
            arm_joint_positions_rad=(np.array([0.2, -0.1]), np.array([0.19, -0.09])),
            gripper_positions_rad=(np.array([0.2]), np.array([0.19])),
        ),
    ]
    settings = ControllerTrackingSettings(period_s=1.0, cycles=1, ramp_s=0.0, settle_s=1.0)
    csv_path = tmp_path / "tracking.csv"
    svg_path = tmp_path / "tracking.svg"

    write_tracking_csv(csv_path, samples)
    write_tracking_svg(svg_path, samples, settings, "triskel")

    with csv_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == len(samples)
    assert rows[-1]["segment"] == "settle"
    assert float(rows[1]["joint_command_timestamp_s"]) == pytest.approx(100.98)
    assert float(rows[1]["state_minus_joint_command_s"]) == pytest.approx(0.02)
    assert float(rows[1]["arm_1_controller_command_rad"]) == pytest.approx(0.4)
    assert float(rows[1]["arm_1_measured_rad"]) == pytest.approx(0.35)
    assert float(rows[1]["arm_1_error_rad"]) == pytest.approx(0.05)
    assert float(rows[1]["position_error_m"]) == pytest.approx(0.002)
    assert float(rows[1]["gripper_1_error_rad"]) == pytest.approx(0.05)

    svg = svg_path.read_text(encoding="utf-8")
    assert "Controller tracking measurement" in svg
    assert "commanded" in svg
    assert "measured" in svg
    assert "Controller output vs measured joints" in svg
    assert "arm_1 error" in svg
    assert ">arm_1</text>" in svg
    assert "joint position (deg)" in svg
    assert svg.count('class="joint-command"') == EXPECTED_ARM_JOINT_PLOTS + 1
    assert svg.count('class="joint-measured"') == EXPECTED_ARM_JOINT_PLOTS + 1
    assert 'class="joint-error"' not in svg
    assert "Gripper 1 error" in svg
    assert "RMS" in svg
