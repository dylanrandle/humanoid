import mujoco
import numpy as np
import pinocchio as pin

from humanoid.config import ROBOT_CONFIGS
from humanoid.robots.base import Robot
from humanoid.types.homing import HomingPreset
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotBaseCommand, RobotName, RobotState, RobotToolCommand
from humanoid.types.simulation import SimulationScene
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
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom * 2 + 1)
    draw_commands(scene, world, RobotBaseCommand(0, pin.SE3.Identity()))
    assert scene.ngeom == len(world.joint_geoms) + len(world.tool_geoms) + 1
    assert scene.geoms[scene.ngeom - 1].type == mujoco.mjtGeom.mjGEOM_ARROW
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
