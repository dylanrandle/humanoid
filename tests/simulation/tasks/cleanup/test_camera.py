"""Four-camera placement, physical mounts, and rendered calibration."""

import os
from io import BytesIO

import mujoco
import numpy as np
import pinocchio as pin
import pytest
from PIL import Image

from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.config.tasks.cleanup import (
    BASE_CAMERA_YAWS,
    TRISKEL_CLEANUP_CAMERAS,
    TRISKEL_WRIST_CAMERA,
    WRIST_CAMERA_NAME,
)
from humanoid.simulation.camera import SimulationCamera
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.simulation.tasks.cleanup.task import CleanupTask


def test_rgb_camera_uses_last_link_mounting_holes_behind_gripper():
    engine = CleanupTask().create_engine(0)
    camera = engine.model.camera(WRIST_CAMERA_NAME)
    assert camera.bodyid == engine.model.body("arm_6_link").id
    assert camera.pos == pytest.approx(TRISKEL_WRIST_CAMERA.position)
    assert camera.pos[2] > 0
    assert engine.model.geom("camera_wrist_housing").bodyid == camera.bodyid
    assert engine.model.geom("camera_wrist_bracket").bodyid == camera.bodyid
    assert engine.model.geom("camera_wrist_lens").bodyid == camera.bodyid
    mount = TRISKEL_WRIST_CAMERA.mount
    assert mount is not None
    assert np.linalg.norm(np.subtract(mount.holes[1], mount.holes[0])) == pytest.approx(
        0.0205, abs=1e-6
    )
    for index, hole in enumerate(mount.holes):
        screw = engine.model.geom(f"camera_wrist_screw_{index}")
        assert screw.bodyid == camera.bodyid
        assert screw.pos - np.asarray(mount.rotation)[
            :, 1
        ] * mount.screw_length / 2 == pytest.approx(hole)


def test_camera_mass_is_added_to_the_robot_without_replacing_urdf_inertia():
    robot = NativeMujocoEngine(TRISKEL_CONFIG)
    engine = CleanupTask().create_engine(0)
    config = TRISKEL_WRIST_CAMERA
    assert config.parent_frame is not None and config.mount is not None
    original = robot.model.body(config.parent_frame)
    mounted = engine.model.body(config.parent_frame)
    added_mass = (
        engine.model.body_subtreemass[mounted.id] - robot.model.body_subtreemass[original.id]
    )
    assert added_mass == pytest.approx(config.housing_mass + config.mount.mass)
    assert mounted.inertia != pytest.approx(original.inertia)
    base = engine.model.body("base_link").id
    original_base = robot.model.body("base_link").id
    assert engine.model.body_subtreemass[base] - robot.model.body_subtreemass[
        original_base
    ] == pytest.approx(
        sum(
            c.housing_mass + (c.mount.mass if c.mount else 0)
            for c in TRISKEL_CLEANUP_CAMERAS.values()
        )
    )


def test_four_cameras_include_outward_views_between_each_pair_of_wheels():
    task = CleanupTask()
    engine = task.create_engine(0)
    names = {engine.model.camera(i).name for i in range(engine.model.ncam)}
    assert names == {"wrist", "base_1_2", "base_2_3", "base_3_1"}
    assert names == set(task.cameras())
    scene_geoms = {geom.name for geom in engine.read_scene_state(0).geoms}
    for name in names:
        assert f"camera_{name}_housing" in scene_geoms
    for name in BASE_CAMERA_YAWS:
        camera = engine.model.camera(name)
        assert camera.bodyid == engine.model.body("base_link").id
        assert np.linalg.norm(camera.pos[:2]) == pytest.approx(0.108)
        config = task.cameras()[name]
        assert config.parent_frame == "lower_plate_link"
        assert camera.pos[2] == pytest.approx(0.029)
        assert config.housing_size is not None and config.mount is not None
        half = np.asarray(config.housing_size) / 2
        axes = np.asarray(config.rotation)
        housing_center = np.asarray(config.position) - axes[:, 2] * half[2]
        vertical_extent = np.abs(axes[2]) @ half
        # The housing fits in the 46 mm gap, with its bracket on the lower plate.
        assert housing_center[2] - vertical_extent > 0.007  # noqa: PLR2004
        assert housing_center[2] + vertical_extent < 0.053  # noqa: PLR2004
        mount = config.mount
        assert mount.position[2] - mount.size[2] / 2 == pytest.approx(0.007)
        forward = np.asarray(config.rotation)[:, 2]
        assert forward[:2] / np.linalg.norm(forward[:2]) == pytest.approx(
            camera.pos[:2] / np.linalg.norm(camera.pos[:2])
        )
        assert forward[2] == pytest.approx(0.0)
        assert np.linalg.det(config.rotation) == pytest.approx(1)
        _, first, second = name.split("_")
        wheels = [engine.data.body(f"wheel_{number}_link").xpos[:2] for number in (first, second)]
        bisector = sum(wheel / np.linalg.norm(wheel) for wheel in wheels)
        assert camera.pos[:2] / np.linalg.norm(camera.pos[:2]) == pytest.approx(
            bisector / np.linalg.norm(bisector), abs=1e-5
        )


def test_base_cameras_follow_base_motion_and_are_independent_of_arm_motion():
    engine = CleanupTask().create_engine(0)
    initial = {name: engine.data.camera(name).xpos.copy() for name in BASE_CAMERA_YAWS}
    engine.data.qpos[int(engine.model.joint("arm_6").qposadr[0])] += 0.15
    mujoco.mj_forward(engine.model, engine.data)
    for name, position in initial.items():
        assert engine.data.camera(name).xpos == pytest.approx(position)
    root = engine.binding.root
    assert root is not None
    engine.data.qpos[list(root.qpos_addresses)] = [0.2, -0.1, np.pi / 2]
    mujoco.mj_forward(engine.model, engine.data)
    rotation = pin.exp3(np.array([0.0, 0.0, np.pi / 2]))
    for name, position in initial.items():
        assert engine.data.camera(name).xpos == pytest.approx(rotation @ position + [0.2, -0.1, 0])


def test_gripper_roll_does_not_move_camera_but_last_arm_joint_does():
    engine = CleanupTask().create_engine(0)
    camera = engine.data.camera(WRIST_CAMERA_NAME)
    initial_position, initial_rotation = camera.xpos.copy(), camera.xmat.copy()
    engine.data.qpos[int(engine.model.joint("arm_7").qposadr[0])] += 0.8
    mujoco.mj_forward(engine.model, engine.data)
    assert camera.xpos == pytest.approx(initial_position)
    assert camera.xmat == pytest.approx(initial_rotation)
    engine.data.qpos[int(engine.model.joint("arm_6").qposadr[0])] += 0.15
    mujoco.mj_forward(engine.model, engine.data)
    assert camera.xpos != pytest.approx(initial_position)
    assert camera.xmat != pytest.approx(initial_rotation)


@pytest.mark.skipif(
    os.environ.get("HUMANOID_RUN_RENDER_TESTS") != "1", reason="Needs MuJoCo graphics."
)
@pytest.mark.parametrize("name", TRISKEL_CLEANUP_CAMERAS)
def test_rgb_calibration_matches_mounted_optical_frame(name):
    engine = CleanupTask().create_engine(0)
    config = TRISKEL_CLEANUP_CAMERAS[name]
    sensor = SimulationCamera(engine, name, config)
    try:
        timestamp = 12.5
        observation = sensor.capture(timestamp)
        frame = observation.frame
        assert frame.timestamp == timestamp
        assert frame.name == name
        assert frame.frame_id == f"{name}_optical"
        assert config.parent_frame is not None
        parent = engine.robot.get_frame_pose(
            config.parent_frame, engine.read_robot_state(0).joint_positions
        )
        expected = parent * pin.SE3(np.asarray(config.rotation), np.asarray(config.position))
        assert observation.world_from_optical.translation == pytest.approx(
            expected.translation, abs=1e-6
        )
        assert observation.world_from_optical.rotation == pytest.approx(expected.rotation, abs=1e-6)
        assert frame.intrinsics[0, 0] == pytest.approx(config.focal_x)
        with Image.open(BytesIO(frame.rgb_jpeg)) as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert image.size == (config.width, config.height)
            assert np.ptp(np.asarray(image)) > 0
    finally:
        sensor.close()
