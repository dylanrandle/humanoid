from dataclasses import replace

import mujoco
import numpy as np
import pytest

from humanoid.config import DEFAULT_MUJOCO_SIMULATION_CONFIG, ROBOT_CONFIGS
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.mirror import SimulationMirror
from humanoid.types.lcm import simulation_state_t
from humanoid.types.lcm.converter import LCMConverter
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
