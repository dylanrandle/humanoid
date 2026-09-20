"""Effort telemetry, joint mapping, and diagnostic reference-line regression tests."""

import csv
from dataclasses import replace
from xml.etree import ElementTree

import numpy as np
import pytest

from humanoid.config.robot.panda import PANDA_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.robots.base import Robot
from humanoid.robots.utils.controller_tracking.effort import (
    actuator_effort_limits,
    build_effort_traces,
)
from humanoid.robots.utils.controller_tracking.report import (
    _append_actuator_effort_grid,
    write_native_joint_telemetry_csv,
)
from humanoid.robots.utils.controller_tracking.timing import _native_joint_sample
from humanoid.types.actuator import ActuatorEffortLimits, ActuatorEffortSource
from humanoid.types.controller_tracking import ActuatorEffortTrace
from humanoid.types.homing import HomingPreset
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import RobotState


@pytest.mark.parametrize("config", [TRISKEL_CONFIG, PANDA_CONFIG], ids=lambda config: config.name)
def test_native_effort_survives_transport_joint_mapping_and_csv(config, tmp_path):
    robot = Robot(config)
    efforts = np.arange(robot.model.nv, dtype=float) * -0.1
    efforts[0] = np.nan
    state = RobotState(
        timestamp=9.0,
        joint_positions=config.homing_presets[HomingPreset.HOME].copy(),
        joint_velocities=np.zeros(robot.model.nv),
        joint_efforts=efforts,
        actuator_temperatures=np.zeros(len(config.actuator_control_modes)),
        effort_source=ActuatorEffortSource.CURRENT_ESTIMATE,
    )
    message = LCMConverter.robot_state_to_lcm(state)
    recovered = LCMConverter.robot_state_from_lcm(type(message).decode(message.encode()))
    sample = _native_joint_sample(
        robot=robot,
        segment="figure_eight",
        setting="xz_1x",
        window_index=0,
        stream="state",
        message=recovered,
        received_at_s=10.0,
    )
    indices = robot.get_joint_velocity_indices(
        [
            *robot.get_arm_joint_indices(),
            *robot.get_gripper_joint_indices(),
        ]
    )
    assert sample.joint_efforts is not None
    np.testing.assert_allclose(sample.joint_efforts, efforts[indices])
    missing = replace(sample, received_timestamp_s=11.0, joint_efforts=None)
    settled = replace(sample, received_timestamp_s=12.0, segment="figure_eight_settle")
    command = replace(sample, stream="controller", joint_efforts=np.full(len(indices), 999.0))
    traces = build_effort_traces([settled, command, missing, sample], config)
    trace = traces["xz_1x"]
    np.testing.assert_allclose(trace.times_s, [10.0, 11.0, 12.0])
    np.testing.assert_allclose(trace.efforts[0], efforts[indices])
    assert np.isnan(trace.efforts[1]).all()
    assert trace.segment_boundaries_s == (12.0,)
    assert "estimated from current" in trace.source
    assert trace.units[-1] == ("N" if config is PANDA_CONFIG else "N·m")
    # Captured feedback owns its data even if the incoming message is reused.
    assert recovered.joint_efforts is not None
    recovered.joint_efforts[:] = 999.0
    np.testing.assert_allclose(sample.joint_efforts, efforts[indices])

    path = tmp_path / "native.csv"
    write_native_joint_telemetry_csv(path, [sample, missing])
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    last_joint = rows[len(indices) - 1]
    assert float(last_joint["effort_si"]) == pytest.approx(efforts[indices[-1]])
    assert last_joint["effort_source"] == "current_estimate"
    assert all(row["effort_si"] == "" for row in rows[len(indices) :])


def test_effort_plot_displays_ratings_and_overruns_without_joining_missing_data():
    trace = ActuatorEffortTrace(
        joint_names=("arm<1>", "gripper"),
        times_s=np.arange(6, dtype=float),
        efforts=np.array(
            [
                [0.3, np.nan],
                [-4.0, np.nan],
                [np.nan, np.nan],
                [0.9, np.nan],
                [1.0, np.nan],
                [np.nan, np.nan],
            ]
        ),
        units=("N·m", "N"),
        source="Synthetic test feedback",
        segment_boundaries_s=(3.0,),
    )
    elements = []
    original_efforts = trace.efforts.copy()
    _append_actuator_effort_grid(
        elements,
        trace=trace,
        heading_y=100.0,
        limits={
            "arm<1>": ActuatorEffortLimits(stall=3.0, rated=1.0),
            "gripper": ActuatorEffortLimits(stall=10.0, rated=5.0, unit="N"),
        },
    )
    svg = "<svg>" + "".join(elements) + "</svg>"
    root = ElementTree.fromstring(svg)
    paths = root.findall("./polyline[@class='effort']")
    expected_runs = 2
    assert len(paths) == expected_runs
    peak_y = float(paths[0].attrib["points"].split()[1].split(",")[1])
    stall_y = float(root.findall("./line[@class='effort-stall']")[0].attrib["y1"])
    rated_y = float(root.findall("./line[@class='effort-rated']")[0].attrib["y1"])
    assert peak_y < stall_y < rated_y
    np.testing.assert_allclose(trace.efforts, original_efforts)
    assert "Maximum (stall): 3 N·m" in svg
    assert "Rated: 1 N·m" in svg
    assert "Rated: 5 N" in svg
    assert "effort magnitude (N)" in svg
    assert "Effort data unavailable." in svg
    assert "nan" not in svg


def test_effort_limits_are_per_actuator_and_do_not_invent_unconfigured_ratings():
    assert actuator_effort_limits(PANDA_CONFIG) == {}
    assert TRISKEL_CONFIG.hardware is not None
    assert TRISKEL_CONFIG.hardware.actuators is not None
    hardware = TRISKEL_CONFIG.hardware.actuators
    limits = ActuatorEffortLimits(stall=1.5, rated=0.4)
    config = replace(
        TRISKEL_CONFIG,
        hardware=replace(
            TRISKEL_CONFIG.hardware,
            actuators=replace(
                hardware,
                joints={
                    **hardware.joints,
                    "arm_2": replace(hardware.joints["arm_2"], effort_limits=limits),
                    "gripper_1": replace(hardware.joints["gripper_1"], effort_limits=None),
                },
            ),
        ),
    )
    reported = actuator_effort_limits(config)
    assert reported["arm_2"] == limits
    assert "gripper_1" not in reported
