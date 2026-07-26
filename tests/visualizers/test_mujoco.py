"""Lifecycle tests for the passive native MuJoCo visualizer."""

from unittest.mock import MagicMock

import mujoco
import pytest

from humanoid.visualizers.mujoco import (
    MujocoSimulationVisualizer,
    relaunch_with_mjpython_on_macos,
)


def _model_and_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    return model, mujoco.MjData(model)


def test_visualizer_launches_passively_and_syncs_while_open(monkeypatch):
    model, data = _model_and_data()
    handle = MagicMock()
    handle.is_running.return_value = True
    launch = MagicMock(return_value=handle)
    monkeypatch.setattr("humanoid.visualizers.mujoco.mujoco.viewer.launch_passive", launch)
    visualizer = MujocoSimulationVisualizer(model, data)

    visualizer.initialize()
    visualizer.sync()
    visualizer.close()

    launch.assert_called_once_with(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    )
    handle.sync.assert_called_once_with()
    handle.close.assert_called_once_with()


def test_visualizer_ignores_sync_after_the_window_closes(monkeypatch):
    model, data = _model_and_data()
    handle = MagicMock()
    handle.is_running.return_value = False
    monkeypatch.setattr(
        "humanoid.visualizers.mujoco.mujoco.viewer.launch_passive",
        MagicMock(return_value=handle),
    )
    visualizer = MujocoSimulationVisualizer(model, data)

    visualizer.initialize()
    visualizer.sync()

    handle.sync.assert_not_called()


def test_visualizer_requires_exactly_one_initialization(monkeypatch):
    model, data = _model_and_data()
    monkeypatch.setattr(
        "humanoid.visualizers.mujoco.mujoco.viewer.launch_passive",
        MagicMock(return_value=MagicMock()),
    )
    visualizer = MujocoSimulationVisualizer(model, data)

    with pytest.raises(RuntimeError, match="not initialized"):
        visualizer.sync()

    visualizer.initialize()
    with pytest.raises(RuntimeError, match="already initialized"):
        visualizer.initialize()


def test_visualizer_skips_native_window_without_a_graphical_display(monkeypatch):
    model, data = _model_and_data()
    launch = MagicMock()
    monkeypatch.setattr("humanoid.visualizers.mujoco.sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("humanoid.visualizers.mujoco.mujoco.viewer.launch_passive", launch)
    visualizer = MujocoSimulationVisualizer(model, data)

    visualizer.initialize()
    visualizer.sync()
    visualizer.close()

    launch.assert_not_called()


def test_visualizer_treats_native_window_failure_as_nonfatal(monkeypatch):
    model, data = _model_and_data()
    monkeypatch.setattr("humanoid.visualizers.mujoco.sys.platform", "darwin")
    monkeypatch.setattr(
        "humanoid.visualizers.mujoco.mujoco.viewer.launch_passive",
        MagicMock(side_effect=RuntimeError("viewer unavailable")),
    )
    visualizer = MujocoSimulationVisualizer(model, data)

    visualizer.initialize()
    visualizer.sync()
    visualizer.close()


def test_macos_relaunches_the_simulation_module_with_mjpython(monkeypatch):
    executable = "/virtual/venv/bin/mjpython"
    execv = MagicMock()
    monkeypatch.setattr("humanoid.visualizers.mujoco.sys.platform", "darwin")
    monkeypatch.delenv("MJPYTHON_BIN", raising=False)
    monkeypatch.setattr(
        "humanoid.visualizers.mujoco.shutil.which", MagicMock(return_value=executable)
    )
    monkeypatch.setattr("humanoid.visualizers.mujoco.os.execv", execv)

    relaunch_with_mjpython_on_macos("humanoid.nodes.robot.simulation")

    execv.assert_called_once_with(
        executable,
        [executable, "-m", "humanoid.nodes.robot.simulation"],
    )


def test_macos_relaunch_requires_mjpython(monkeypatch):
    monkeypatch.setattr("humanoid.visualizers.mujoco.sys.platform", "darwin")
    monkeypatch.delenv("MJPYTHON_BIN", raising=False)
    monkeypatch.setattr("humanoid.visualizers.mujoco.shutil.which", MagicMock(return_value=None))

    with pytest.raises(RuntimeError, match="requires the mjpython executable"):
        relaunch_with_mjpython_on_macos("humanoid.nodes.robot.simulation")
