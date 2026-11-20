"""Read-only trial through the dashboard, native sensor encoding and RAX backend."""

import base64
import io
import json
import time
from dataclasses import replace
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from humanoid.config.robot.triskel import TRISKEL_CONFIG
from humanoid.constants import Topic
from humanoid.integrations.rax import benchmark, contact_preview
from humanoid.integrations.rax.tools import CONTACT_PREVIEW_TASK
from humanoid.orchestrator.monitor import observation
from humanoid.orchestrator.service import OrchestratorService
from humanoid.robots.base import Robot
from humanoid.types.camera import CameraEncoding, CameraImage, CameraIntrinsics
from humanoid.types.homing import HomingPreset
from humanoid.types.orchestrator import OrchestratorError
from humanoid.types.robot import RobotState
from humanoid.ui.server import create_app


@pytest.fixture
def sensors(monkeypatch):
    robot = Robot(TRISKEL_CONFIG)
    now = time.perf_counter()
    camera = CameraIntrinsics(
        width=32,
        height=24,
        fx=20,
        fy=20,
        cx=16,
        cy=12,
        distortion_model="none",
        distortion_coefficients=(),
    )
    rgb = CameraImage(
        timestamp=now,
        device_timestamp=12,
        timestamp_domain="device",
        frame_number=453,
        serial_number="d435-test",
        encoding=CameraEncoding.RGB8,
        frame_id="ego_camera_color_optical_frame",
        data=np.zeros((24, 32, 3), dtype=np.uint8),
        intrinsics=camera,
    )
    depth = replace(
        rgb,
        frame_number=450,
        encoding=CameraEncoding.DEPTH16,
        frame_id="ego_camera_depth_optical_frame",
        depth_scale=0.001,
        data=np.full((24, 32), 456, dtype=np.uint16),
    )
    q = TRISKEL_CONFIG.homing_presets[HomingPreset.HOME]
    state = RobotState(
        timestamp=now,
        joint_positions=q,
        joint_velocities=np.zeros(robot.model.nv),
        actuator_temperatures=np.zeros(robot.model.nv),
    )
    messages = {Topic.CAMERA_RGB: rgb, Topic.CAMERA_DEPTH: depth, Topic.ROBOT_STATE: state}
    subscriber = Mock()
    subscriber.receive.side_effect = lambda topic, timeout: messages[topic]
    monkeypatch.setattr(observation, "Subscriber", Mock(return_value=subscriber))
    return robot, messages, subscriber


def test_live_trial_exports_frame_linked_contacts_without_control(sensors, tmp_path, monkeypatch):
    robot, messages, subscriber = sensors
    service = Mock(spec=OrchestratorService)
    service.hardware_observation.side_effect = lambda: observation.hardware_snapshot(robot)
    client = create_app(service).test_client()

    def get(request, timeout):
        assert request.method == "GET" and request.data is None
        assert request.selector == "/api/hardware/observation"
        result = client.get(request.selector)
        assert result.status_code == HTTPStatus.OK
        return io.BytesIO(result.data)

    monkeypatch.setattr(contact_preview, "build_opener", lambda *_: SimpleNamespace(open=get))
    session = benchmark.backend(SimpleNamespace(task=CONTACT_PREVIEW_TASK, root=tmp_path))
    assert {t["name"] for t in session.tools()} == {
        "humanoid_status",
        "humanoid_observe",
        "humanoid_mark_contacts",
    }
    first, _ = session.start(formal=False, seed=0)
    current, images = session.call("humanoid_observe", {})
    assert current["state"]["tool_frame"] == "gripper_base_link"
    assert first["observation_id"] != current["observation_id"]
    assert [a["content_index"] for a in current["attachments"]] == [1, 2]
    assert current["attachments"][1]["depth_scale"] == messages[Topic.CAMERA_DEPTH].depth_scale
    assert (
        current["attachments"][0]["optical_to_base"] != current["attachments"][1]["optical_to_base"]
    )
    assert not session.evaluate()["task_complete"]
    for name in ("humanoid_move_tool", "humanoid_move_base", "humanoid_gripper", "humanoid_hold"):
        with pytest.raises(ValueError, match="Unsupported"):
            session.call(name, {})
    report = {"observation_id": current["observation_id"], "contacts": [[8, 8], [22, 8]]}
    with pytest.raises(ValueError, match="latest"):
        session.call(
            "humanoid_mark_contacts", {**report, "observation_id": first["observation_id"]}
        )
    with pytest.raises(ValueError, match="in-bounds"):
        session.call("humanoid_mark_contacts", {**report, "contacts": [[8, 8], [32, 8]]})
    session.call("humanoid_mark_contacts", report)
    directory = tmp_path / "contact-previews" / current["episode_id"]
    assert (directory / "rgb.png").read_bytes() == base64.b64decode(images[0]["data"])
    assert (directory / "depth.png").read_bytes() == base64.b64decode(images[1]["data"])
    record = json.loads((directory / "contacts.json").read_text())
    assert record["frames"]["rgb"]["frame_number"] == messages[Topic.CAMERA_RGB].frame_number
    assert record["contacts_rgb_px"] == report["contacts"]
    with Image.open(directory / "contacts.png") as overlay:
        assert overlay.getpixel((8, 3)) == (0, 255, 255)
        assert overlay.getpixel((22, 3)) == (255, 0, 255)
    assert session.evaluate()["task_complete"]
    session.close()
    assert subscriber.close.call_count == service.hardware_observation.call_count
    assert service.method_calls == [("hardware_observation", (), {})] * 2
    with pytest.raises(RuntimeError, match="Start"):
        session.call("humanoid_observe", {})


@pytest.mark.parametrize(
    "problem", ["unpaired", "serial", "simulation", "stale_state", "stale_frames"]
)
def test_hardware_snapshot_rejects_unusable_observations(sensors, problem, monkeypatch):
    robot, messages, subscriber = sensors
    rgb = messages[Topic.CAMERA_RGB]
    if problem == "unpaired":
        messages[Topic.CAMERA_RGB] = replace(rgb, timestamp=rgb.timestamp - 0.001)
    elif problem == "serial":
        messages[Topic.CAMERA_RGB] = replace(rgb, serial_number="different")
    elif problem == "simulation":
        for topic in (Topic.CAMERA_RGB, Topic.CAMERA_DEPTH):
            messages[topic] = replace(messages[topic], serial_number="simulation")
    elif problem == "stale_frames":
        for topic in (Topic.CAMERA_RGB, Topic.CAMERA_DEPTH):
            messages[topic] = replace(messages[topic], timestamp=rgb.timestamp - 1)
    else:
        messages[Topic.ROBOT_STATE] = replace(
            messages[Topic.ROBOT_STATE], timestamp=rgb.timestamp - 1
        )
    monkeypatch.setattr(
        observation,
        "time",
        SimpleNamespace(perf_counter=Mock(side_effect=[rgb.timestamp] * 3 + [rgb.timestamp + 3])),
    )
    with pytest.raises(OrchestratorError, match="fresh paired"):
        observation.hardware_snapshot(robot)
    subscriber.close.assert_called_once()


@pytest.mark.parametrize("task", [CONTACT_PREVIEW_TASK, "toy_car_cleanup"])
def test_live_discovery_matches_backend_tools(task, monkeypatch, tmp_path):
    monkeypatch.setenv("HUMANOID_RAX_TASK", task)
    monkeypatch.setenv("HUMANOID_RAX_TOKEN", "test-only")
    config = SimpleNamespace(task=task, root=tmp_path)
    discovered = benchmark.tool_definitions()
    assert discovered == benchmark.backend(config).tools()
    assert benchmark.registration(SimpleNamespace).robot_tools == {t["name"] for t in discovered}
    names = {t["name"] for t in discovered}
    assert ("humanoid_mark_contacts" in names) == (task == CONTACT_PREVIEW_TASK)
    assert ("humanoid_move_tool" in names) == (task == "toy_car_cleanup")
