import os
import sys
import time
from contextlib import nullcontext
from unittest.mock import Mock

import lcm
import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.constants import Topic
from humanoid.types.homing import HomingPreset
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.process import Runtime
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotName,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.simulation import SimulationScene, SimulationState
from humanoid.visualizers import cli, stream
from humanoid.visualizers.telemetry import LiveTelemetry


@pytest.mark.parametrize("remote", [False, True], ids=["local", "ssh"])
@pytest.mark.parametrize("runtime", list(Runtime))
def test_existing_state_and_command_topics(tmp_path, monkeypatch, runtime, remote):
    install = tmp_path / "remote install"
    install.mkdir()
    (install / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    ssh = tmp_path / "ssh"
    ssh.write_text('#!/bin/sh\nfor arg do remote="$arg"; done\nexec /bin/sh -c "$remote"\n')
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("HUMANOID_ROBOT_DIR", str(install))

    robot = RobotName.TRISKEL
    q = ROBOT_CONFIGS[robot].homing_presets[HomingPreset.HOME]
    log_path = tmp_path / "states.lcm"
    event_log = lcm.EventLog(log_path, "w")
    event_log.write_event(0, "IGNORED", b"")
    for i, (topic, packet) in enumerate(
        [
            (
                Topic.ROBOT_JOINT_COMMAND,
                LCMConverter.robot_joint_command_to_lcm(RobotJointCommand(0, q + 1)),
            ),
            (
                Topic.ROBOT_TOOL_COMMAND,
                LCMConverter.robot_tool_command_to_lcm(
                    RobotToolCommand(0, pin.SE3.Identity(), np.array([0.1]))
                ),
            ),
            (
                Topic.ROBOT_BASE_COMMAND,
                LCMConverter.robot_base_command_to_lcm(RobotBaseCommand(0, pin.SE3.Identity())),
            ),
        ]
    ):
        event_log.write_event(100_000 + 10_000 * i, topic.value, packet.encode())
    # Large integration vectors exercise packet reassembly across pipe reads.
    physics = np.arange(12000.0)
    if runtime is Runtime.SIM:
        scene = SimulationScene.TOY_CLEANUP
        packet = LCMConverter.simulation_state_to_lcm(SimulationState(robot, scene, physics))
        topic = Topic.SIMULATION_STATE
    else:
        scene = SimulationScene.EMPTY
        packet = LCMConverter.robot_state_to_lcm(RobotState(1, q, np.zeros(1), np.zeros(1)))
        topic = Topic.ROBOT_STATE
    event_log.write_event(150_000, topic.value, packet.encode())
    event_log.write_event(60_000_000, "IGNORED", b"")
    event_log.close()
    url = f"file://{log_path}?mode=r"
    monkeypatch.setenv("LCM_DEFAULT_URL", url)
    monkeypatch.setattr(stream, "DEFAULT_LCM_URL", url)
    with stream.ssh_frames("triskel") if remote else stream.local_frames() as receive:
        deadline = time.monotonic() + 10
        view = LiveTelemetry()
        while view.state is None and time.monotonic() < deadline:
            view.update(receive(100))
        assert view.model_key == (robot, runtime, scene)
        if runtime is Runtime.SIM:
            assert isinstance(view.state, SimulationState)
            np.testing.assert_array_equal(view.state.state, physics)
        else:
            assert isinstance(view.state, RobotState)
            np.testing.assert_array_equal(view.state.joint_positions, q)
        np.testing.assert_array_equal(view.joint_target, q + 1)
        assert view.tool is not None and view.base is not None
        np.testing.assert_array_equal(view.tool.gripper_positions, [0.1])


@pytest.mark.parametrize("host", [None, "triskel", "jetson"])
def test_viewer_selects_ssh_only_with_explicit_host(monkeypatch, host):
    monkeypatch.setenv("HUMANOID_ROBOT_HOST", "unrelated")
    monkeypatch.setenv("MJPYTHON_BIN", "test")
    monkeypatch.setattr(cli, "local_frames", lambda: nullcontext("local"))
    monkeypatch.setattr(cli, "ssh_frames", nullcontext)
    show = Mock()
    monkeypatch.setattr(cli, "show", show)
    cli.run(["--host", host] if host else [])
    show.assert_called_once_with(host or "local", host or "localhost")
