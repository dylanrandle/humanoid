"""Display live telemetry in MuJoCo's passive native viewer."""

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from humanoid.config import ROBOT_CONFIG, ROBOT_CONFIGS, VISUALIZER_CONFIG
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotBaseCommand, RobotState
from humanoid.types.simulation import SimulationScene
from humanoid.visualizers.stream import Receive
from humanoid.visualizers.telemetry import LiveTelemetry
from humanoid.visualizers.world import ViewerWorld

_REFRESH_SECONDS = 1 / 60


def draw_commands(
    scene: mujoco.MjvScene, world: ViewerWorld, base: RobotBaseCommand | None
) -> None:
    """Populate only the user scene so native visualization options remain independent."""
    scene.ngeom = 0
    model = world.model
    for visible, indices, poses, color in (
        (
            world.show_joints,
            world.joint_geoms,
            world.joint_ghost,
            [1, 0.55, 0.12, VISUALIZER_CONFIG.joint_command_opacity],
        ),
        (
            world.show_tool,
            world.tool_geoms,
            world.tool_ghost,
            [0.05, 0.65, 1, VISUALIZER_CONFIG.tool_command_opacity],
        ),
    ):
        if not visible:
            continue
        rgba = np.asarray(color, dtype=np.float32)
        for i in indices:
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            kind = model.geom_type[i]
            mujoco.mjv_initGeom(
                geom, kind, model.geom_size[i], poses.geom_xpos[i], poses.geom_xmat[i], rgba
            )
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                # MuJoCo uses the even render ID for the mesh, the odd ID for its hull.
                geom.dataid = 2 * model.geom_dataid[i]
            scene.ngeom += 1
    if (
        base is None
        or not VISUALIZER_CONFIG.show_commanded_base_pose
        or scene.ngeom >= scene.maxgeom
    ):
        return
    if not np.isfinite(base.pose.homogeneous).all():
        raise ValueError("Invalid base command pose.")
    arrow = scene.geoms[scene.ngeom]
    kind = mujoco.mjtGeom.mjGEOM_ARROW
    mujoco.mjv_initGeom(
        arrow,
        kind,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.array([1, 0.7, 0.1, 1], dtype=np.float32),
    )
    mujoco.mjv_connector(
        arrow, kind, 0.012, base.pose.translation, base.pose.act(np.array([0.16, 0, 0]))
    )
    scene.ngeom += 1


def show(receive: Receive, host: str) -> None:
    live = LiveTelemetry()
    live.update(receive(100))
    while True:
        key = live.model_key or (ROBOT_CONFIG.name, Runtime.REAL, SimulationScene.EMPTY)
        name, runtime, scene = key
        robot = Robot(ROBOT_CONFIGS[name])
        world = ViewerWorld(robot, scene, runtime)
        if live.state is None:
            q = robot.config.homing_presets[HomingPreset.HOME]
            world.apply(RobotState(0, q, np.zeros(robot.model.nv), np.empty(0)), None, None)
        else:
            world.apply(live.state, live.joint_target, live.tool)
        with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
            previous_label = None
            while viewer.is_running():
                deadline = time.monotonic() + _REFRESH_SECONDS
                changed = False
                while (remaining := deadline - time.monotonic()) > 0:
                    if live.update(receive(math.ceil(remaining * 1000))):
                        changed = True
                if live.model_key is not None and live.model_key != key:
                    break  # The native viewer binds its model at launch.
                with viewer.lock():
                    if changed and live.state is not None:
                        world.apply(live.state, live.joint_target, live.tool)
                    draw_commands(viewer.user_scn, world, live.base)
                label = f"{host} · {name} · {runtime} · {scene}"
                if live.model_key is None:
                    label = f"{host} · {name} · Home pose · Waiting for state"
                elif time.monotonic() - live.last_received > 1:
                    label += " · Waiting for fresh state"
                if label != previous_label:
                    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, label, ""))
                    previous_label = label
                viewer.sync()
            else:
                return
