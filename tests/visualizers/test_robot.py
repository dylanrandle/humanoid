import subprocess
from unittest.mock import MagicMock, call

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset
from humanoid.types.visualizer import VisualizerConfig
from humanoid.visualizers.robot import (
    MESHCAT_SERVER_STOP_TIMEOUT_SECONDS,
    RobotVisualizer,
    ToolCommandVisualizer,
)


def _visualizer_with_window(server_process):
    visualizer = RobotVisualizer(MagicMock(), VisualizerConfig())
    window = MagicMock()
    window.server_proc = server_process
    viewer = MagicMock()
    viewer.viewer.window = window
    visualizer._viewer = viewer
    visualizer._initialized = True
    return visualizer, window


def test_close_stops_owned_meshcat_server_and_is_idempotent():
    server_process = MagicMock()
    server_process.poll.return_value = None
    visualizer, window = _visualizer_with_window(server_process)

    visualizer.close()
    visualizer.close()

    window.zmq_socket.close.assert_called_once_with(linger=0)
    server_process.terminate.assert_called_once_with()
    server_process.wait.assert_called_once_with(timeout=MESHCAT_SERVER_STOP_TIMEOUT_SECONDS)
    server_process.kill.assert_not_called()
    assert visualizer._viewer is None
    assert visualizer._initialized is False


def test_close_kills_meshcat_server_that_does_not_terminate():
    server_process = MagicMock()
    server_process.poll.return_value = None
    server_process.wait.side_effect = [
        subprocess.TimeoutExpired("meshcat", MESHCAT_SERVER_STOP_TIMEOUT_SECONDS),
        0,
    ]
    visualizer, _ = _visualizer_with_window(server_process)

    visualizer.close()

    server_process.kill.assert_called_once_with()
    assert server_process.wait.call_args_list == [
        call(timeout=MESHCAT_SERVER_STOP_TIMEOUT_SECONDS),
        call(),
    ]


def test_close_leaves_an_externally_managed_meshcat_server_running():
    visualizer, window = _visualizer_with_window(server_process=None)

    visualizer.close()

    window.zmq_socket.close.assert_called_once_with(linger=0)


def _initialized_tool_visualizer() -> tuple[ToolCommandVisualizer, MagicMock, Robot]:
    robot = Robot(TRISKEL_CONFIG)
    visualizer = ToolCommandVisualizer(robot, MagicMock(), robot.config.tool.frame)
    viewer = MagicMock()
    visualizer._viewer = viewer
    visualizer._initialized = True
    visualizer._reference_q = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME].copy()
    visualizer._ee_pose_at_reference = pin.SE3.Identity()
    return visualizer, viewer, robot


def test_tool_command_visualizer_applies_bounded_gripper_position():
    visualizer, viewer, robot = _initialized_tool_visualizer()
    gripper_index = robot.get_gripper_position_indices()[0]
    _, upper_limit = robot.get_gripper_limits()[0]

    visualizer.display(pin.SE3.Identity(), np.array([upper_limit + 1.0]))

    displayed_q = viewer.display.call_args.args[0]
    assert displayed_q[gripper_index] == pytest.approx(upper_limit)


@pytest.mark.parametrize("gripper_positions", [np.array([]), np.array([np.nan])])
def test_tool_command_visualizer_rejects_invalid_gripper_target(gripper_positions):
    visualizer, viewer, _ = _initialized_tool_visualizer()

    with pytest.raises(ValueError, match="Gripper target"):
        visualizer.display(pin.SE3.Identity(), gripper_positions)

    viewer.display.assert_not_called()
