"""The empty task preserves the original robot-only simulation."""

import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.simulation.tasks.empty import EmptySceneTask
from humanoid.types.homing import HomingPreset


@pytest.mark.parametrize("robot_config", ROBOT_CONFIGS.values(), ids=ROBOT_CONFIGS)
def test_empty_scene_uses_selected_robot_without_task_geometry(robot_config):
    task = EmptySceneTask(robot_config)
    engine = task.create_engine(0)

    assert engine.robot.config.name is robot_config.name
    assert engine.read_robot_state(0).joint_positions == pytest.approx(
        robot_config.homing_presets[HomingPreset.HOME]
    )
    assert engine.read_scene_state(0).geoms == ()
    assert engine.model.ncam == 0
    assert task.objects(engine) == ()
    assert task.cameras() == {}
