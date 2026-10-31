"""Render simulation-only scene telemetry in the existing MeshCat viewer."""

from typing import Any

import numpy as np
import pinocchio as pin
from meshcat import geometry

from humanoid.types.scene import SimulationSceneState


class SceneVisualizer:
    def __init__(self, viewer: Any):
        self.viewer = viewer
        self.viewer["simulation"].delete()
        self.signatures: dict[str, tuple] = {}

    def display(self, scene: SimulationSceneState) -> None:
        names = {geom.name for geom in scene.geoms}
        for name in self.signatures.keys() - names:
            self.viewer[f"simulation/{name}"].delete()
            del self.signatures[name]
        for geom in scene.geoms:
            node = self.viewer[f"simulation/{geom.name}"]
            signature = (geom.kind, tuple(geom.size), tuple(geom.rgba))
            if self.signatures.get(geom.name) != signature:
                if geom.kind == "box":
                    shape = geometry.Box(2 * geom.size)
                elif geom.kind == "cylinder":
                    shape = geometry.Cylinder(2 * geom.size[1], geom.size[0])
                else:
                    shape = geometry.Box([2, 2, 0.002])
                red, green, blue = (255 * geom.rgba[:3]).astype(int)
                color = int((red << 16) | (green << 8) | blue)
                node.set_object(shape, geometry.MeshLambertMaterial(color=color))
                self.signatures[geom.name] = signature
            pose = geom.pose.copy()
            if geom.kind == "cylinder":
                pose.rotation = pose.rotation @ pin.exp3(np.array([np.pi / 2, 0, 0]))
            node.set_transform(pose.homogeneous)
