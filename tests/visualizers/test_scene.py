"""Simulation primitives are updated without reloading meshes every frame."""

from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pinocchio as pin

from humanoid.types.scene import SceneGeometry, SimulationSceneState
from humanoid.visualizers.scene import SceneVisualizer


def test_reuses_geometry_and_removes_absent_objects():
    viewer = MagicMock()
    nodes = defaultdict(MagicMock)
    viewer.__getitem__.side_effect = nodes.__getitem__
    visualizer = SceneVisualizer(viewer)
    viewer["simulation"].delete.assert_called_once()
    geom = SceneGeometry("toy", "box", np.ones(3), np.ones(4), pin.SE3.Identity())
    visualizer.display(SimulationSceneState(0, (geom,)))
    visualizer.display(SimulationSceneState(1, (geom,)))
    viewer["simulation/toy"].set_object.assert_called_once()
    assert viewer["simulation/toy"].set_transform.call_count == 2  # noqa: PLR2004
    visualizer.display(SimulationSceneState(2, ()))
    viewer["simulation/toy"].delete.assert_called_once()
