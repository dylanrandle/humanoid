"""Native MuJoCo viewer for the simulation engine's model and live data."""

from __future__ import annotations

import os
import shutil
import sys

import mujoco
import mujoco.viewer

from humanoid.logger import get_logger

_DARWIN_PLATFORM = "darwin"
_LINUX_PLATFORM = "linux"
_MJPYTHON_ENVIRONMENT_VARIABLE = "MJPYTHON_BIN"
_GRAPHICAL_DISPLAY_ENVIRONMENT_VARIABLES = ("DISPLAY", "WAYLAND_DISPLAY")

logger = get_logger(__name__)


class MujocoSimulationVisualizer:
    """Own a passive MuJoCo viewer synchronized by the simulation loop."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._model = model
        self._data = data
        self._handle: mujoco.viewer.Handle | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Open the native viewer without giving it ownership of physics stepping."""
        if self._initialized:
            raise RuntimeError("MuJoCo simulation visualizer is already initialized.")
        self._initialized = True
        if not _graphical_display_available():
            logger.warning(
                "Native MuJoCo viewer disabled because no graphical display is available"
            )
            return
        try:
            self._handle = mujoco.viewer.launch_passive(
                self._model,
                self._data,
                show_left_ui=False,
                show_right_ui=False,
            )
        except (mujoco.FatalError, RuntimeError) as exc:
            logger.warning("Native MuJoCo viewer could not start: %s", exc)

    def sync(self) -> None:
        """Copy the latest physics state into the open viewer."""
        self._require_initialized()
        if self._handle is not None and self._handle.is_running():
            handle = self._handle
            handle.sync()

    def close(self) -> None:
        """Close the viewer if it was initialized."""
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MuJoCo simulation visualizer is not initialized.")


def _graphical_display_available() -> bool:
    if sys.platform != _LINUX_PLATFORM:
        return True
    return any(os.getenv(name) for name in _GRAPHICAL_DISPLAY_ENVIRONMENT_VARIABLES)


def relaunch_with_mjpython_on_macos(module_name: str) -> None:
    """Re-exec a simulation entrypoint under ``mjpython`` when macOS requires it."""
    if sys.platform != _DARWIN_PLATFORM or os.getenv(_MJPYTHON_ENVIRONMENT_VARIABLE):
        return
    executable = shutil.which("mjpython")
    if executable is None:
        raise RuntimeError("MuJoCo visualization on macOS requires the mjpython executable.")
    os.execv(executable, [executable, "-m", module_name])
