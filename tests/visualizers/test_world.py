from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.robots.base import Robot
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.homing import HomingPreset
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotName, RobotState, RobotToolCommand
from humanoid.types.simulation import MujocoSimulationConfig, SimulationScene
from humanoid.visualizers.world import ViewerWorld


def test_empty_simulation_has_a_reference_floor_without_changing_playback():
    engine = NativeMujocoEngine(
        ROBOT_CONFIGS[RobotName.TRISKEL], MujocoSimulationConfig(scene=SimulationScene.EMPTY)
    )
    try:
        engine.step(20)
        world = ViewerWorld(engine.robot, SimulationScene.EMPTY, Runtime.SIM)
        world.apply(engine.simulation_state(), None, None)
        floor = world.model.geom("reference_floor")
        assert floor.contype == 0 and floor.conaffinity == 0
        wheels = engine.robot.config.wheels
        assert wheels
        home = engine.robot.config.homing_presets[HomingPreset.HOME]
        assert world.data.geom("reference_floor").xpos[2] == pytest.approx(
            engine.robot.get_frame_pose(wheels[0].floor_frame, home).translation[2]
        )
        np.testing.assert_array_equal(world.data.qpos, engine.data.qpos)
        np.testing.assert_array_equal(world.data.qvel, engine.data.qvel)
    finally:
        engine.close()


@pytest.mark.parametrize("name", list(RobotName))
def test_simulation_preserves_live_scene_and_robot(name):
    scene = SimulationScene.TOY_CLEANUP
    engine = NativeMujocoEngine(ROBOT_CONFIGS[name], MujocoSimulationConfig(scene=scene))
    try:
        address = engine.model.jnt_qposadr[engine.model.body("ball_0").jntadr[0]]
        engine.data.qpos[address] += 0.25
        engine.step(20)
        q = engine.read_robot_state(0).joint_positions
        world = ViewerWorld(engine.robot, scene, Runtime.SIM)
        model = world.model
        root = engine.robot.get_root_q_slice()
        if root is not None:
            q[root] = [2, 1, 0, 1]
        world.apply(engine.simulation_state(), q, RobotToolCommand(0, pin.SE3.Identity()))
        if world.binding.root is not None:
            addresses = list(world.binding.root.qpos_addresses)
            np.testing.assert_allclose(
                world.joint_ghost.qpos[addresses], engine.data.qpos[addresses]
            )
        np.testing.assert_array_equal(world.data.qpos, engine.data.qpos)
        np.testing.assert_array_equal(world.data.qvel, engine.data.qvel)
        assert world.data.time == engine.data.time
        assert model.nbody == engine.model.nbody
        np.testing.assert_array_equal(
            model.geom("task_floor").rgba, engine.model.geom("task_floor").rgba
        )
        assert model.geom("task_floor").matid == engine.model.geom("task_floor").matid
        assert len(world.joint_geoms) and len(world.tool_geoms)
    finally:
        engine.close()


def test_hardware_measured_base_gripper_and_modeled_optics():
    robot = Robot(ROBOT_CONFIGS[RobotName.TRISKEL])
    q = robot.config.homing_presets[HomingPreset.HOME].copy()
    root = robot.get_root_q_slice()
    assert root is not None
    q[root] = [0.5, -0.2, 0, 1]
    joint_target = q.copy()
    joint_target[root] = [2, 1, 1, 0]
    joint_target[robot.get_gripper_position_indices()] = 0.2
    wrist = robot.config.cameras["wrist"].hardware
    assert wrist is not None
    tool = RobotToolCommand(0, pin.SE3.Identity(), np.array([0.2]))
    state = RobotState(0, q, np.zeros(robot.model.nv), np.zeros(0))
    world = ViewerWorld(robot, SimulationScene.EMPTY, Runtime.REAL)
    model = world.model
    world.apply(state, joint_target, tool)
    assert world.binding.root is not None
    np.testing.assert_allclose(
        world.data.qpos[list(world.binding.root.qpos_addresses)], [0.5, -0.2, np.pi / 2]
    )
    camera_id = model.camera(wrist.rgb_frame).id
    optics = robot.config.cameras["wrist"].simulation
    assert optics is not None
    fov = 2 * np.rad2deg(
        np.arctan(model.cam_sensorsize[camera_id] / (2 * model.cam_intrinsic[camera_id, :2]))
    )
    np.testing.assert_allclose(fov, optics.rgb_fov_degrees, atol=1e-4)
    np.testing.assert_allclose(
        world.data.cam_xpos[camera_id],
        robot.get_frame_pose(wrist.rgb_frame, q).translation,
        atol=1e-6,
    )
    assert world.show_joints and world.show_tool
    joint_ids, tool_ids = world.joint_geoms, world.tool_geoms
    wrist_positions = world.tool_ghost.geom_xpos[tool_ids].copy()
    joints = world.joint_ghost.geom_xpos[joint_ids].copy()
    np.testing.assert_allclose(
        world.joint_ghost.qpos[list(world.binding.root.qpos_addresses)], [0.5, -0.2, np.pi / 2]
    )
    np.testing.assert_allclose(world.joint_ghost.joint("gripper_1").qpos, [0.2])
    # Both ghosts follow the measured base even while the command stays fixed.
    moved = q.copy()
    moved[root.start] += 0.4
    world.apply(replace(state, joint_positions=moved), joint_target, tool)
    np.testing.assert_allclose(
        world.tool_ghost.geom_xpos[tool_ids], wrist_positions + np.array([0.4, 0, 0]), atol=1e-8
    )
    np.testing.assert_allclose(
        world.joint_ghost.geom_xpos[joint_ids], joints + np.array([0.4, 0, 0]), atol=1e-8
    )
