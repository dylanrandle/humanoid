from dataclasses import replace

import mujoco
import numpy as np
import pytest

from humanoid.config.tasks import TOY_CAR_CLEANUP_CONFIG
from humanoid.tasks.toy_car_cleanup import ToyCarCleanupTask


@pytest.fixture
def scene():
    task = ToyCarCleanupTask(replace(TOY_CAR_CLEANUP_CONFIG, car_count=1))
    spec = mujoco.MjSpec()
    task.build_scene(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    task.reset(model, data, seed=7)
    return task, model, data


def place_car(scene, xyz):
    _, model, data = scene
    address = model.joint("car_0_free").qposadr[0]
    data.qpos[address : address + 7] = [*xyz, 1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def test_seed_reproducible_and_changes_layout(scene):
    task, model, data = scene
    original = data.qpos.copy()
    task.reset(model, data, seed=8)
    assert not np.array_equal(original, data.qpos)
    task.reset(model, data, seed=7)
    np.testing.assert_array_equal(original, data.qpos)


def test_cars_settle_on_floor_without_success(scene):
    task, model, data = scene
    for _ in range(1000):
        mujoco.mj_step(model, data)
        result = task.evaluate(model, data)
    assert np.isfinite(data.qpos).all()
    assert not result.task_complete
    assert data.xpos[model.body("car_0").id, 2] == pytest.approx(
        task.config.ground_z + task.config.car_height / 2, abs=0.002
    )


def test_bin_position_alone_cannot_pass(scene):
    task, model, data = scene
    c = task.config
    place_car(scene, [*c.bin_center, c.ground_z + c.bin_wall_thickness + c.car_height / 2])
    task.evaluate(model, data)
    data.time += 2
    assert not task.evaluate(model, data).task_complete


def test_release_settling_and_containment_required(scene, monkeypatch):
    task, model, data = scene
    c = task.config
    # Oracle fixture: isolate prior robot-contact lift from deposit geometry.
    task._lifted[0] = True
    place_car(scene, [*c.bin_center, c.ground_z + c.bin_wall_thickness + c.car_height / 2])
    assert not task.evaluate(model, data).task_complete
    data.time += c.settle_seconds
    assert task.evaluate(model, data).task_complete
    place_car(scene, [c.bin_center[0] + c.bin_inner_size[0] / 2, c.bin_center[1], 0.05])
    assert not task.evaluate(model, data).task_complete


def test_held_car_in_bin_does_not_pass(scene, monkeypatch):
    task, model, data = scene
    c = task.config
    task._lifted[0] = True
    place_car(scene, [*c.bin_center, c.ground_z + c.bin_wall_thickness + c.car_height / 2 - 0.001])
    assert data.ncon > 0
    monkeypatch.setattr(task, "_is_robot_contact", lambda *_: True)
    task.evaluate(model, data)
    data.time += c.settle_seconds * 2
    assert not task.evaluate(model, data).task_complete


def test_reset_clears_evaluation_history(scene):
    task, model, data = scene
    task._lifted[0] = True
    task._multiple_carry = True
    task.reset(model, data, seed=7)
    assert not any(task._lifted)
    assert task.evaluate(model, data).reason == "cars_remaining"
