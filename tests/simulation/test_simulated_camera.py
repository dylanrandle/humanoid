"""URDF camera placement, native depth semantics, and simulation timing."""

from dataclasses import replace
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest

from humanoid.config.camera import SIMULATED_CAMERA_CONFIG
from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.simulation.camera import MujocoCamera
from humanoid.simulation.engine import NativeMujocoEngine
from humanoid.types.camera import CameraEncoding

COLLISION_GROUP = 3
EXPECTED_RENDERED_PAIRS = 3


def test_camera_poses_follow_urdf_at_nonzero_mobile_pose():
    engine = NativeMujocoEngine(TRISKEL_CONFIG)
    assert set(engine.cameras) == {"base", "wrist"}
    root = engine.binding.root
    assert root is not None
    engine.data.qpos[list(root.qpos_addresses)] = [0.4, -0.2, 0.7]
    mujoco.mj_forward(engine.model, engine.data)
    q = engine.read_robot_state(1.0).joint_positions
    for frame in (
        frame
        for camera in engine.cameras.values()
        for frame in (camera.config.rgb_frame, camera.config.depth_frame)
    ):
        expected = engine.robot.get_frame_pose(frame, q).copy()
        camera_id = engine.model.camera(frame).id
        np.testing.assert_allclose(engine.data.cam_xpos[camera_id], expected.translation, atol=1e-8)
        np.testing.assert_allclose(
            engine.data.cam_xmat[camera_id].reshape(3, 3),
            expected.rotation @ np.diag([1, -1, -1]),
            atol=1e-8,
        )
    assert np.any(engine.model.geom_group == COLLISION_GROUP)
    visual = engine.model.geom_group != COLLISION_GROUP
    assert np.all(engine.model.geom_contype[visual] == 0)
    engine.close()


def test_native_rendering_depth_conversion_cadence_and_cleanup(monkeypatch):
    config = replace(SIMULATED_CAMERA_CONFIG, width=4, height=2)
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <camera name="{config.rgb_frame}" fovy="42"/>
      <camera name="{config.depth_frame}" fovy="58"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    renderer = MagicMock()
    factory = MagicMock(return_value=renderer)
    monkeypatch.setattr("humanoid.simulation.camera.mujoco.Renderer", factory)
    rgb = np.full((2, 4, 3), 12, dtype=np.uint8)
    meters = np.array([[1.0, 2.5, 0, np.nan], [np.inf, 1000, -1, 0.001]], dtype=np.float32)
    registered_meters = np.flip(meters, axis=1).copy()
    renderer.render.side_effect = [rgb, registered_meters, meters] * EXPECTED_RENDERED_PAIRS
    camera = MujocoCamera(model, config)
    factory.assert_not_called()
    frames = camera.read(data, timestamp=5.0)
    assert frames is not None
    assert frames.rgb.encoding is CameraEncoding.RGB8
    assert frames.depth.encoding is CameraEncoding.DEPTH16
    np.testing.assert_array_equal(frames.depth.data, [[1000, 2500, 0, 0], [0, 0, 0, 1]])
    assert frames.registered is not None
    np.testing.assert_array_equal(frames.registered.image.data, np.flip(frames.depth.data, axis=1))
    assert frames.registered.image.intrinsics == frames.rgb.intrinsics
    np.testing.assert_array_equal(frames.rgb.data, rgb)
    assert frames.rgb.frame_id == config.rgb_frame
    assert frames.depth.frame_id == config.depth_frame
    assert frames.rgb.intrinsics.fx != frames.depth.intrinsics.fx
    assert frames.depth.timestamp_domain == "simulation_clock"
    assert frames.depth.timestamp == pytest.approx(5.0)
    assert frames.depth.device_timestamp == pytest.approx(0.0)
    assert frames.depth.depth_scale == pytest.approx(config.depth_scale)
    assert [c.kwargs["camera"] for c in renderer.update_scene.call_args_list] == [
        config.rgb_frame,
        config.depth_frame,
    ]
    data.time = 0.03
    assert camera.read(data, 5.03) is None
    data.time = 0.035
    assert camera.read(data, 5.035) is not None
    camera.reset()
    data.time = 0
    assert camera.read(data, 6.0) is not None
    camera.close()
    camera.close()
    renderer.close.assert_called_once()
    assert renderer.disable_depth_rendering.call_count == EXPECTED_RENDERED_PAIRS
