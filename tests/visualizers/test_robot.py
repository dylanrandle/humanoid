import subprocess
from unittest.mock import MagicMock, call

from humanoid.types.visualizer import VisualizerConfig
from humanoid.visualizers.robot import (
    MESHCAT_SERVER_STOP_TIMEOUT_SECONDS,
    RobotVisualizer,
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
