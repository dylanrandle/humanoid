from contextlib import closing
from dataclasses import replace

import mujoco
import numpy as np
import pinocchio as pin
import pytest

from humanoid.config import DEFAULT_MUJOCO_SIMULATION_CONFIG, ROBOT_CONFIGS
from humanoid.config.robot import PANDA_CONFIG, TRISKEL_CONFIG
from humanoid.controllers.omniwheel_base import OmniwheelBaseController
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.mirror import SimulationMirror
from humanoid.simulation.model import ROBOT_CONTACT_TYPE
from humanoid.simulation.tasks.toy_cleanup import (
    TOYS,
    ToyCleanupTask,
    deposited_toys,
    randomize_cleanup,
)
from humanoid.types.lcm import simulation_state_t
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import RobotJointCommand
from humanoid.types.simulation import SimulationScene


@pytest.mark.parametrize("robot_config", ROBOT_CONFIGS.values(), ids=lambda config: config.name)
def test_cleanup_toys_and_viewer_state_round_trip(robot_config):
    config = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=SimulationScene.TOY_CLEANUP)
    engine = NativeMujocoEngine(robot_config, config)
    for kind in ("car", "cube", "ball", "cylinder"):
        for index in range(3):
            body = engine.model.body(f"{kind}_{index}")
            assert engine.model.jnt_type[body.jntadr[0]] == mujoco.mjtJoint.mjJNT_FREE

    ball = engine.model.body("ball_0")
    address = engine.model.jnt_qposadr[ball.jntadr[0]]
    engine.data.qpos[address] += 0.12
    engine.step(100)
    packet = LCMConverter.simulation_state_to_lcm(engine.simulation_state()).encode()
    mirror = SimulationMirror(
        LCMConverter.simulation_state_from_lcm(simulation_state_t.decode(packet))
    )
    np.testing.assert_array_equal(mirror.data.qpos, engine.data.qpos)
    np.testing.assert_array_equal(mirror.data.qvel, engine.data.qvel)
    assert mirror.data.time == engine.data.time
    engine.close()


@pytest.mark.parametrize(
    ("x", "y", "yaw"), [(0, 0.1, 0), (0.1, 0, 0), (0, 0, 0.5)], ids=["forward", "strafe", "yaw"]
)
def test_triskel_can_drive_on_the_cleanup_floor(x: float, y: float, yaw: float):
    config = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=SimulationScene.TOY_CLEANUP)
    with closing(NativeMujocoEngine(TRISKEL_CONFIG, config)) as engine:
        robot = engine.robot
        controller = OmniwheelBaseController(robot, TRISKEL_CONFIG.omniwheel_base_config)
        controller.update_state(engine.read_robot_state(0).joint_positions)
        target = pin.SE3(pin.utils.rotate("z", yaw), np.array([x, y, 0.0]))

        # Run the pose controller from its own commands, as it runs during teleop.
        substeps = round(controller.config.dt / engine.physics_timestep)
        dt = substeps * engine.physics_timestep
        for tick in range(round(1 / dt)):
            result = controller.compute_control(target, dt=dt)
            engine.apply_joint_command(
                RobotJointCommand(
                    timestamp=tick * dt, joint_positions=result.q, joint_velocities=result.v
                )
            )
            engine.step(substeps)

        measured = robot.get_base_pose(engine.read_robot_state(1).joint_positions)
        assert measured is not None
        np.testing.assert_allclose(measured.translation, target.translation, atol=0.01)
        np.testing.assert_allclose(measured.rotation, target.rotation, atol=0.02)


def test_randomized_cleanup_reset_and_scoring():
    config = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=SimulationScene.TOY_CLEANUP)

    def layout(seed):
        engine = NativeMujocoEngine(
            TRISKEL_CONFIG, config, configure_scene=lambda spec: randomize_cleanup(spec, seed)
        )
        try:
            data = engine.data
            positions = np.stack([data.body(name).xpos.copy() for name in TOYS])
            assert np.all(positions[:, 0] > data.geom("wall_left").xpos[0])
            assert np.all(positions[:, 0] < data.geom("wall_right").xpos[0])
            assert np.all(positions[:, 1] < data.geom("wall_front").xpos[1])
            assert not deposited_toys(data)
            body = engine.model.body("cube_0")
            address = engine.model.jnt_qposadr[body.jntadr[0]]
            data.qpos[address : address + 3] = data.geom("bin_floor").xpos + np.array([0, 0, 0.03])
            engine.step(round(0.1 / engine.physics_timestep))
            assert deposited_toys(data) == ["cube_0"]
            engine.reset()
            np.testing.assert_array_equal(data.body("cube_0").xpos, positions[TOYS.index("cube_0")])
            return [
                positions,
                data.geom("bin_floor").xpos.copy(),
                data.geom("wall_right").xpos.copy(),
                *[
                    engine.model.geom(name).rgba.copy()
                    for name in ("chassis_0", "task_floor", "wall_left", "bin_floor")
                ],
            ]
        finally:
            engine.close()

    original = layout(12)
    for a, b, c in zip(original, layout(12), layout(13), strict=True):
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)


def test_single_item_episode_compiles_only_the_cube_and_scores_its_deposit():
    toys = ("cube_0",)
    config = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=SimulationScene.TOY_CLEANUP)
    engine = NativeMujocoEngine(
        TRISKEL_CONFIG, config, configure_scene=lambda spec: randomize_cleanup(spec, 0, toys)
    )
    try:
        present = [
            name
            for name in TOYS
            if mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        ]
        assert present == list(toys)
        assert deposited_toys(engine.data, toys) == []
        joint = engine.model.body("cube_0").jntadr[0]
        address = engine.model.jnt_qposadr[joint]
        engine.data.qpos[address : address + 3] = engine.data.geom("bin_floor").xpos + np.array(
            [0, 0, 0.03]
        )
        engine.step(round(0.125 / engine.physics_timestep))
        assert deposited_toys(engine.data, toys) == list(toys)
    finally:
        engine.close()


def test_deposit_requires_release_and_support_including_stacked_toys():
    task = ToyCleanupTask(("cube_0", "cube_1", "car_0"))
    config = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=task.scene)

    def configure(spec):
        task.configure_scene(spec, 0)
        finger = spec.worldbody.add_body(name="test_finger", mocap=True, pos=[0, 0, 1])
        finger.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01] * 3, contype=ROBOT_CONTACT_TYPE)

    with closing(NativeMujocoEngine(TRISKEL_CONFIG, config, configure_scene=configure)) as engine:
        model, data = engine.model, engine.data
        floor = data.geom("bin_floor").xpos.copy()
        for name, offset in (
            ("cube_0", [0, 0, 0.05]),
            ("cube_1", [0, 0, 0.09]),
            ("car_0", [0.07, 0, 0.04]),
        ):
            address = model.jnt_qposadr[model.body(name).jntadr[0]]
            data.qpos[address : address + 3] = floor + offset
        mujoco.mj_forward(model, data)
        assert task.score(data).metrics["deposited"] == []
        engine.step(round(0.6 / engine.physics_timestep))
        assert task.score(data).success

        # The upper cube still has support from below while the finger presses on it.
        mocap = model.body("test_finger").mocapid[0]
        data.mocap_pos[mocap] = data.body("cube_1").xpos + np.array([0, 0, 0.027])
        mujoco.mj_forward(model, data)
        assert task.score(data).metrics["deposited"] == ["cube_0", "car_0"]
        data.mocap_pos[mocap, 2] += 0.1
        engine.step(round(0.6 / engine.physics_timestep))
        assert task.score(data).success


@pytest.mark.parametrize("seed", [0, 5])
def test_panda_cube_layout(seed):
    task = ToyCleanupTask(robot=PANDA_CONFIG.name)
    config = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=task.scene)
    with closing(
        NativeMujocoEngine(
            PANDA_CONFIG, config, configure_scene=lambda spec: task.configure_scene(spec, seed)
        )
    ) as engine:
        data = engine.data
        cube = data.body("cube_0").xpos
        bin_center = data.geom("bin_floor").xpos
        for center in (cube, bin_center):
            assert 0.26 <= center[0] <= 0.34  # noqa: PLR2004
            assert 0.14 <= abs(center[1]) <= 0.23  # noqa: PLR2004
        assert cube[1] * bin_center[1] < 0
        assert not engine.cameras
