"""Viewer telemetry preserves shapes/poses and remains a separate LCM topic."""

import numpy as np
import pinocchio as pin
import pytest

from humanoid.constants import TOPIC_TO_TYPE, Topic
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.lcm.simulation_scene_state_t import simulation_scene_state_t
from humanoid.types.scene import SceneGeometry, SimulationSceneState


def test_scene_roundtrip_preserves_geometry_and_pose():
    geom = SceneGeometry(
        "block",
        "box",
        np.array([0.02, 0.03, 0.04]),
        np.array([1, 0, 0, 1]),
        pin.SE3(pin.exp3(np.array([0, 0, 0.6])), np.array([0.1, 0.2, 0.3])),
    )
    original = SimulationSceneState(1.25, (geom,))
    encoded = LCMConverter.simulation_scene_to_lcm(original).encode()
    restored = LCMConverter.simulation_scene_from_lcm(simulation_scene_state_t.decode(encoded))
    assert restored.timestamp == original.timestamp
    assert restored.geoms[0].name == geom.name
    assert restored.geoms[0].kind == geom.kind
    assert restored.geoms[0].size == pytest.approx(geom.size)
    assert restored.geoms[0].pose.homogeneous == pytest.approx(geom.pose.homogeneous)
    assert TOPIC_TO_TYPE[Topic.SIMULATION_SCENE] is SimulationSceneState


def test_empty_scene_can_clear_previous_visualization():
    message = LCMConverter.simulation_scene_to_lcm(SimulationSceneState(0, ()))
    assert (
        LCMConverter.simulation_scene_from_lcm(
            simulation_scene_state_t.decode(message.encode())
        ).geoms
        == ()
    )
