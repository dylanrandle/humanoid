"""Visibility diagnostics report occlusion without constraining collection."""

import os

import mujoco
import numpy as np
import pytest

from humanoid.config.tasks.cleanup import WRIST_CAMERA_NAME
from humanoid.simulation.tasks.cleanup.motion import CleanupMotion
from humanoid.simulation.tasks.cleanup.task import CleanupTask
from humanoid.simulation.tasks.cleanup.visibility import CleanupView


@pytest.fixture
def visible_scene(stack_runtime):
    task = CleanupTask()
    runtime = stack_runtime(task)
    engine = runtime.engine
    motion = CleanupMotion(
        runtime, task.arm_config, task.mobile_config, task.scene, task.view_config
    )
    motion.park()
    task.motion = motion
    # Controlled visibility fixtures, not autonomous collection state changes.
    address = int(engine.model.joint("red_block_free").qposadr[0])
    camera = engine.data.camera(WRIST_CAMERA_NAME)
    forward = -camera.xmat.reshape(3, 3)[:, 2]
    distance = (task.scene.floor_height + 0.013 - camera.xpos[2]) / forward[2]
    engine.data.qpos[address : address + 3] = camera.xpos + forward * distance
    mujoco.mj_forward(engine.model, engine.data)
    return task, engine, runtime, motion


def occlude_red(engine):
    camera = engine.data.camera(WRIST_CAMERA_NAME).xpos
    red = engine.data.body("red_block").xpos
    address = int(engine.model.joint("orange_car_free").qposadr[0])
    engine.data.qpos[address : address + 3] = (camera + red) / 2
    mujoco.mj_forward(engine.model, engine.data)


def test_visibility_requires_real_line_of_sight_not_just_projection(visible_scene):
    _, engine, _, motion = visible_scene
    before = motion.view.measure("red_block")
    assert before.in_frame and before.visible
    occlude_red(engine)
    after = motion.view.measure("red_block")
    assert after.in_frame
    assert after.box_pixels == pytest.approx(before.box_pixels)
    assert not after.visible
    assert after.visible_fraction == 0


def test_base_view_keeps_target_observable_when_wrist_view_is_occluded(visible_scene):
    task, engine, runtime, motion = visible_scene
    motion.view_target = "red_block"
    occlude_red(engine)
    assert not motion.view.measure("red_block").visible
    assert motion.views()["red_block"].camera == "base_2_3"
    assert task.observation_errors(runtime) == ()


def test_target_hidden_from_all_cameras_does_not_stop_policy_commands(visible_scene):
    task, engine, runtime, motion = visible_scene
    motion.view_target = "red_block"
    occlude_red(engine)
    engine.model.geom("orange_car_shape").size[:] = [0.2, 0.1, 0.4]
    mujoco.mj_forward(engine.model, engine.data)
    assert not motion.views()["red_block"].visible
    assert task.observation_errors(runtime) == ()
    initial_count = runtime.action_count
    motion.hold(0.2)
    assert runtime.action_count > initial_count


@pytest.mark.skipif(
    os.environ.get("HUMANOID_RUN_RENDER_TESTS") != "1", reason="Needs MuJoCo graphics."
)
def test_visibility_rays_agree_with_actual_camera_rasterization(visible_scene):
    task, engine, _, motion = visible_scene
    config = task.scene.cameras[WRIST_CAMERA_NAME]
    renderer = mujoco.Renderer(engine.model, height=config.height, width=config.width)
    renderer.enable_segmentation_rendering()
    try:
        for occluded in (False, True):
            if occluded:
                occlude_red(engine)
            renderer.update_scene(engine.data, camera=WRIST_CAMERA_NAME)
            segmentation = renderer.render()
            pixels = np.count_nonzero(
                (segmentation[:, :, 0] == engine.model.geom("red_block_shape").id)
                & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
            )
            measured = motion.view.measure("red_block")
            assert measured.visible == (not occluded)
            if occluded:
                assert pixels == 0
            else:
                assert pixels > task.view_config.min_box_pixels
                assert measured.box_pixels * measured.visible_fraction == pytest.approx(
                    pixels, rel=0.25
                )
    finally:
        renderer.close()


@pytest.mark.skipif(
    os.environ.get("HUMANOID_RUN_RENDER_TESTS") != "1", reason="Needs MuJoCo graphics."
)
@pytest.mark.parametrize("camera_name", ["base_2_3", "base_3_1"])
def test_bin_crossing_near_plane_still_matches_rendered_rgb_visibility(camera_name):
    task = CleanupTask()
    engine = task.create_engine(0)
    root = engine.binding.root
    assert root is not None
    engine.data.qpos[list(root.qpos_addresses)] = [
        task.scene.bin_position[0],
        task.scene.bin_position[1] + 0.24,
        2 * np.pi / 3,
    ]
    mujoco.mj_forward(engine.model, engine.data)
    view = CleanupView(engine, task.scene, task.view_config)
    camera = engine.data.camera(camera_name)
    points = (view.bin_corners - camera.xpos) @ (camera.xmat.reshape(3, 3) @ np.diag([1, -1, -1]))
    near = engine.model.vis.map.znear * engine.model.stat.extent
    assert points[:, 2].min() < near < points[:, 2].max()
    config = task.cameras()[camera_name]
    renderer = mujoco.Renderer(engine.model, height=config.height, width=config.width)
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(engine.data, camera=camera_name)
        segmentation = renderer.render()
        margin = int(task.view_config.frame_margin)
        segmentation = segmentation[margin:-margin, margin:-margin]
        target_ids = [engine.model.geom(f"bin_{index}").id for index in range(5)]
        pixels = np.count_nonzero(
            np.isin(segmentation[:, :, 0], target_ids)
            & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
        )
        measured = view.measure("cleanup_bin", camera_name=camera_name)
        assert measured.visible and not measured.in_frame
        assert measured.box_pixels * measured.visible_fraction == pytest.approx(pixels, rel=0.25)
    finally:
        renderer.close()
