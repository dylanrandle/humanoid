import json

import pytest

from humanoid.config import ROBOT_CONFIGS
from humanoid.recording import (
    DEFAULT_RECORDING_ROOT,
    RECORDING_LOG_FILENAME,
    RECORDING_MANIFEST_FILENAME,
    RECORDING_SCHEMA_VERSION,
    RecordingCatalog,
    RecordingError,
    serialize_robot_config,
)
from humanoid.types.robot import RobotName
from humanoid.utils.paths import find_repo_root

EXPECTED_GRIPPER_ACTUATOR_ID = 8


def test_default_recording_root_is_anchored_to_repository():
    assert find_repo_root(__file__) / "logs" == DEFAULT_RECORDING_ROOT


def test_create_writes_robot_config_beside_exact_log_path(tmp_path):
    catalog = RecordingCatalog(tmp_path)
    config = ROBOT_CONFIGS[RobotName.PANDA]

    recording = catalog.create(config)

    assert recording.directory.parent == tmp_path
    assert recording.log_path == recording.directory / RECORDING_LOG_FILENAME
    assert recording.manifest_path == recording.directory / RECORDING_MANIFEST_FILENAME
    manifest = json.loads(recording.manifest_path.read_text())
    assert manifest["recording_id"] == recording.id
    assert manifest["schema_version"] == RECORDING_SCHEMA_VERSION
    assert manifest["robot"] == RobotName.PANDA
    assert manifest["robot_config"] == serialize_robot_config(config)


def test_list_returns_only_complete_managed_recordings(tmp_path):
    catalog = RecordingCatalog(tmp_path)
    first = catalog.create(ROBOT_CONFIGS[RobotName.PANDA])
    first.log_path.write_bytes(b"log")
    incomplete = catalog.create(ROBOT_CONFIGS[RobotName.SO101])
    (tmp_path / "unrelated").mkdir()

    summaries = catalog.list()

    assert len(summaries) == 1
    assert summaries[0].id == first.id
    assert summaries[0].robot is RobotName.PANDA
    assert summaries[0].id != incomplete.id


def test_get_rejects_paths_outside_recording_root(tmp_path):
    catalog = RecordingCatalog(tmp_path)

    with pytest.raises(RecordingError, match="valid recording"):
        catalog.get("../recording_elsewhere")


def test_get_rejects_manifest_that_does_not_match_directory(tmp_path):
    catalog = RecordingCatalog(tmp_path)
    recording = catalog.create(ROBOT_CONFIGS[RobotName.PANDA])
    manifest = json.loads(recording.manifest_path.read_text())
    manifest["recording_id"] = "recording_different"
    recording.manifest_path.write_text(json.dumps(manifest))
    recording.log_path.write_bytes(b"log")

    with pytest.raises(RecordingError, match="does not match its directory"):
        catalog.get(recording.id)


def test_serialized_config_distinguishes_equal_sized_robots():
    panda = serialize_robot_config(ROBOT_CONFIGS[RobotName.PANDA])
    elrobot = serialize_robot_config(ROBOT_CONFIGS[RobotName.ELROBOT])
    panda_home = panda["home_position"]
    elrobot_home = elrobot["home_position"]

    assert isinstance(panda_home, list)
    assert isinstance(elrobot_home, list)
    assert len(panda_home) == len(elrobot_home)
    assert panda != elrobot


def test_serialized_config_includes_physical_actuator_details():
    config = serialize_robot_config(ROBOT_CONFIGS[RobotName.ELROBOT_MOBILE])
    hardware = config["hardware"]

    assert isinstance(hardware, dict)
    actuators = hardware["actuators"]
    assert isinstance(actuators, dict)
    controllers = actuators["controllers"]
    assert isinstance(controllers, dict)
    assert "main" in controllers
    joints = actuators["joints"]
    assert isinstance(joints, dict)
    gripper = joints["gripper_1"]
    assert isinstance(gripper, dict)
    assert gripper["actuator_id"] == EXPECTED_GRIPPER_ACTUATOR_ID
    assert gripper["inverted"] is True
    state_estimation = config["state_estimation"]
    assert isinstance(state_estimation, dict)
    root_provider = state_estimation["root"]
    assert root_provider == {}


def test_serialized_panda_config_has_no_physical_hardware():
    config = serialize_robot_config(ROBOT_CONFIGS[RobotName.PANDA])

    assert config["hardware"] is None
