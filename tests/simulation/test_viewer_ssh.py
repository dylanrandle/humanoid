import os
import sys
import time
from pathlib import Path

import lcm
import numpy as np
import pytest

from humanoid.constants import Topic
from humanoid.simulation import viewer
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import RobotName
from humanoid.types.simulation import SimulationScene, SimulationState


def test_remote_relay_streams_state_through_local_shell(tmp_path, monkeypatch):
    # Execute the same remote command in a shell, with a recorded LCM bus as input.
    # A deployment path with spaces also exercises the SSH remote-shell quoting.
    install = tmp_path / "remote install"
    install.mkdir()
    (install / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    ssh = tmp_path / "ssh"
    ssh.write_text('#!/bin/sh\nfor arg do remote="$arg"; done\nexec /bin/sh -c "$remote"\n')
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("HUMANOID_ROBOT_DIR", str(install))

    log_path = tmp_path / "states.lcm"
    event_log = lcm.EventLog(log_path, "w")
    # Make the frame larger than one pipe read to exercise partial frames.
    expected = SimulationState(RobotName.TRISKEL, SimulationScene.TOY_CLEANUP, np.arange(12000.0))
    packet = LCMConverter.simulation_state_to_lcm(expected).encode()
    event_log.write_event(0, Topic.ROBOT_STATE.value, b"ignored")
    event_log.write_event(1, Topic.SIMULATION_STATE.value, packet)
    event_log.write_event(60_000_000, Topic.SIMULATION_STATE.value, packet)
    event_log.close()
    monkeypatch.setenv("LCM_DEFAULT_URL", f"file://{log_path}?mode=r")
    with viewer.ssh_frames("triskel") as receive:
        deadline = time.monotonic() + 5
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = receive(100)
        assert frame is not None
        assert frame.robot == expected.robot
        assert frame.scene == expected.scene
        np.testing.assert_array_equal(frame.state, expected.state)


def test_macos_relaunch_preserves_host(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sim-viewer", "--host", "triskel"])
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("MJPYTHON_BIN", raising=False)
    launcher = Path(sys.executable).with_name("mjpython")

    def capture_exec(path, args):
        assert path == launcher
        assert args == [str(launcher), "-m", "humanoid.simulation.viewer", "--host", "triskel"]
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", capture_exec)
    with pytest.raises(SystemExit) as result:
        viewer.main()
    assert result.value.code == 0
