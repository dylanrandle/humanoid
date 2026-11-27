from unittest.mock import Mock, call

import mujoco
import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotBaseCommand, RobotName, RobotState, RobotToolCommand
from humanoid.types.simulation import SimulationScene
from humanoid.visualizers import native
from humanoid.visualizers.native import draw_commands
from humanoid.visualizers.world import ViewerWorld


def test_commands_use_native_decorations_and_leave_model_cameras_available():
    robot = Robot(ROBOT_CONFIGS[RobotName.TRISKEL])
    world = ViewerWorld(robot, SimulationScene.EMPTY, Runtime.REAL)
    model = world.model
    assert model.geom("reference_floor").id >= 0
    q = robot.config.homing_presets[HomingPreset.HOME].copy()
    state = RobotState(0, q, np.zeros(robot.model.nv), np.empty(0))
    world.apply(state, q, RobotToolCommand(0, pin.SE3.Identity()))
    assert robot.config.wheels
    floor_z = world.data.geom("reference_floor").xpos[2]
    for wheel in robot.config.wheels:
        center_z = robot.get_frame_pose(wheel.frame, q).translation[2]
        assert floor_z == pytest.approx(robot.get_frame_pose(wheel.floor_frame, q).translation[2])
        assert center_z - floor_z == pytest.approx(wheel.radius, abs=1e-6)
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom * 2 + 1)
    pose = pin.SE3(pin.utils.rotate("z", np.pi / 3), np.array([0.3, -0.2, 0.0]))
    draw_commands(scene, world, RobotBaseCommand(0, pose))
    assert scene.ngeom == len(world.joint_geoms) + len(world.tool_geoms) + 1
    arrow = scene.geoms[scene.ngeom - 1]
    assert arrow.type == mujoco.mjtGeom.mjGEOM_ARROW
    mount = robot.get_frame_pose("arm_base_link", q).translation.copy()
    mount[2] = 0
    forward = pose.rotation @ (mount / np.linalg.norm(mount))
    np.testing.assert_allclose(arrow.mat.reshape(3, 3)[:, 2], forward, atol=1e-7)
    for slot, i in enumerate(world.joint_geoms):
        geom = scene.geoms[slot]
        np.testing.assert_allclose(geom.pos, world.joint_ghost.geom_xpos[i], atol=1e-7)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            assert geom.dataid == 2 * model.geom_dataid[i]
    world.apply(state, None, None)
    draw_commands(scene, world, None)
    assert scene.ngeom == 0
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1
    mujoco.mjv_addGeoms(
        model, world.data, options, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DECOR, scene
    )
    assert {
        g.objid for g in scene.geoms[: scene.ngeom] if g.objtype == mujoco.mjtObj.mjOBJ_CAMERA
    } == set(range(model.ncam))


def test_replacement_waits_for_native_window_teardown(monkeypatch):
    world = Mock()
    handle = Mock()
    busy = RuntimeError("another MuJoCo viewer is already open")
    launch = Mock(side_effect=[busy, busy, handle])
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    wait = Mock()
    monkeypatch.setattr(native.time, "sleep", wait)

    assert native._launch_viewer(world, replacing=True) is handle
    wait.assert_has_calls([call(native._REFRESH_SECONDS), call(native._REFRESH_SECONDS)])
    launch.assert_called_with(world.model, world.data)


@pytest.mark.parametrize(
    ("replacing", "error", "elapsed"),
    [
        (False, "another MuJoCo viewer is already open", 0),
        (True, "failed to initialize viewer", 0),
        (True, "another MuJoCo viewer is already open", native._VIEWER_CLOSE_TIMEOUT_SECONDS),
    ],
)
def test_launch_reports_first_window_other_errors_and_timeout(
    monkeypatch, replacing, error, elapsed
):
    launch = Mock(side_effect=RuntimeError(error))
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(native.time, "monotonic", Mock(side_effect=[0, elapsed]))
    with pytest.raises(RuntimeError, match=error):
        native._launch_viewer(Mock(), replacing=replacing)
    launch.assert_called_once()
