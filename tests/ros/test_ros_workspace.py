"""Structural checks for the ROS 2 Triskel workspace.

These tests deliberately avoid importing ROS so the repository can catch description and
configuration drift on non-ROS development machines.
"""

import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast
from xml.etree import ElementTree as ET

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ROS_SRC = REPO_ROOT / "ros_ws" / "src"
DESCRIPTION_PACKAGE = ROS_SRC / "triskel_description"
HARDWARE_PACKAGE = ROS_SRC / "triskel_hardware"
CONTROL_PACKAGE = ROS_SRC / "triskel_control"
MOVEIT_PACKAGE = ROS_SRC / "triskel_moveit_config"
OPERATOR_PACKAGE = ROS_SRC / "triskel_operator"
VISUALIZATION_PACKAGE = ROS_SRC / "triskel_visualization"
DOCKER_DIRECTORY = REPO_ROOT / "docker"
URDF_PATH = DESCRIPTION_PACKAGE / "urdf" / "triskel.urdf"
HARDWARE_XACRO_PATH = HARDWARE_PACKAGE / "urdf" / "triskel.ros2_control.xacro"

EXPECTED_PACKAGES = {
    "triskel_bringup",
    "triskel_control",
    "triskel_description",
    "triskel_hardware",
    "triskel_moveit_config",
    "triskel_operator",
    "triskel_visualization",
}
PACKAGE_URI_PATTERN = re.compile(r"^package://([^/]+)/(.+)$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MAX_COMMAND_TIMEOUT_SECONDS = 0.25
MAX_SMOKE_RUNNER_LINES = 80
VECTOR_DIMENSIONS = 3
WHEEL_RADIUS = 0.05
BASE_GROUND_OFFSET = 0.0321
WHEEL_IDS = [f"wheel_{index}" for index in range(1, 4)]
JOINT_IDS = [f"arm_{index}" for index in range(1, 8)]
GRIPPER_ID = "gripper_1"
GRIPPER_OPEN_POSITION = 0.0
GRIPPER_CLOSED_POSITION = -2.2
EXPECTED_ARM_AXES = {
    "arm_1": (0.0, 0.008727, -0.999962),
    "arm_2": (0.999962, 0.000076, -0.008726),
    "arm_3": (0.999848, 0.000152, -0.017452),
    "arm_4": (0.99981, -0.008574, -0.017527),
    "arm_5": (0.008573, 0.999963, -0.000151),
    "arm_6": (0.999697, -0.0173, -0.017525),
    "arm_7": (0.017144, 0.999812, -0.009029),
}
CALIBRATED_HOME_CONDITION = 41.0
DEFAULT_QUEST_POSE_TOPIC = "/triskel/teleop/meta_quest/right_controller_pose"
DEFAULT_QUEST_JOY_TOPIC = "/triskel/teleop/meta_quest/joy"
MOTOR_IDS = {
    **dict(zip(WHEEL_IDS, range(250, 253), strict=True)),
    **dict(zip(JOINT_IDS, range(1, 8), strict=True)),
    GRIPPER_ID: 8,
}


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict)
    return data


def _local_name(tag: str) -> str:
    return tag.rpartition("}")[2]


def _urdf_joints() -> dict[str, ET.Element]:
    root = ET.parse(URDF_PATH).getroot()
    return {joint.attrib["name"]: joint for joint in root.findall("joint")}


def _docker_smoke_sources() -> tuple[str, list[Path], str]:
    runner = (DOCKER_DIRECTORY / "test_ros2.sh").read_text()
    modules = sorted((DOCKER_DIRECTORY / "tests").glob("*.bash"))
    combined = "\n".join([runner, *(path.read_text() for path in modules)])
    return runner, modules, combined


def test_ros_packages_are_well_formed_and_named_for_their_directories():
    package_dirs = {path.parent.name for path in ROS_SRC.glob("*/package.xml")}
    assert package_dirs == EXPECTED_PACKAGES

    for package_dir in sorted(ROS_SRC.iterdir()):
        if package_dir.name not in EXPECTED_PACKAGES:
            continue
        manifest = ET.parse(package_dir / "package.xml").getroot()
        assert manifest.findtext("name") == package_dir.name
        assert manifest.findtext("version") in {"0.2.0", "0.3.0"}
        assert manifest.findtext("license")
        export = manifest.find("export")
        assert export is not None
        build_type = export.findtext("build_type")
        assert build_type in {"ament_cmake", "ament_python"}
        if build_type == "ament_cmake":
            assert (package_dir / "CMakeLists.txt").is_file()
        else:
            setup_path = package_dir / "setup.py"
            assert setup_path.is_file()
            assert (package_dir / "resource" / package_dir.name).is_file()
            setup_source = setup_path.read_text()
            manifest_version = manifest.findtext("version")
            manifest_description = manifest.findtext("description")
            assert manifest_version is not None
            assert manifest_description is not None
            assert "Path(__file__).resolve().parent" in setup_source
            assert manifest_version not in setup_source
            assert manifest_description not in setup_source


def test_canonical_urdf_meshes_resolve_inside_description_package():
    root = ET.parse(URDF_PATH).getroot()
    assert root.attrib["name"] == "triskel"

    for mesh in root.iter("mesh"):
        match = PACKAGE_URI_PATTERN.fullmatch(mesh.attrib["filename"])
        assert match is not None
        package_name, relative_path = match.groups()
        assert package_name == "triskel_description"
        assert (DESCRIPTION_PACKAGE / relative_path).is_file()

    collisions = list(root.iter("collision"))
    assert collisions
    assert not any(collision.find("geometry/mesh") is not None for collision in collisions)


def test_ros_control_interfaces_match_triskel_hardware_and_urdf_limits():
    root = ET.parse(HARDWARE_XACRO_PATH).getroot()
    declarations = {
        element.attrib["name"]: element
        for element in root.iter()
        if _local_name(element.tag) in {"position_joint", "velocity_joint"}
        and "motor_id" in element.attrib
    }
    expected_joint_names = {*WHEEL_IDS, *JOINT_IDS, GRIPPER_ID}
    assert declarations.keys() == expected_joint_names

    urdf_joints = _urdf_joints()
    for joint_name, declaration in declarations.items():
        assert int(declaration.attrib["motor_id"]) == MOTOR_IDS[joint_name]
        assert joint_name in urdf_joints

        tag = _local_name(declaration.tag)
        if joint_name in WHEEL_IDS:
            assert tag == "velocity_joint"
            continue

        assert tag == "position_joint"
        limit = urdf_joints[joint_name].find("limit")
        assert limit is not None
        assert math.isclose(float(declaration.attrib["lower"]), float(limit.attrib["lower"]))
        assert math.isclose(float(declaration.attrib["upper"]), float(limit.attrib["upper"]))

    position_macro = next(
        element
        for element in root.iter()
        if _local_name(element.tag) == "macro" and element.attrib.get("name") == "position_joint"
    )
    assert position_macro.attrib["params"] == "name motor_id lower upper"

    gripper_limit = urdf_joints[GRIPPER_ID].find("limit")
    gripper_2_mimic = urdf_joints["gripper_2"].find("mimic")
    gripper_3_mimic = urdf_joints["gripper_3"].find("mimic")
    assert gripper_limit is not None
    assert gripper_2_mimic is not None
    assert gripper_3_mimic is not None
    assert math.isclose(float(gripper_limit.attrib["lower"]), GRIPPER_CLOSED_POSITION)
    assert math.isclose(float(gripper_limit.attrib["upper"]), GRIPPER_OPEN_POSITION)
    assert gripper_2_mimic.attrib["multiplier"] == "0.0115"
    assert gripper_3_mimic.attrib["multiplier"] == "-0.0115"


def test_arm_axes_follow_sts_feedback_coordinates():
    urdf_joints = _urdf_joints()
    for joint_name, expected_axis in EXPECTED_ARM_AXES.items():
        axis = urdf_joints[joint_name].find("axis")
        assert axis is not None
        values = tuple(float(value) for value in axis.attrib["xyz"].split())
        assert len(values) == VECTOR_DIMENSIONS
        assert all(
            math.isclose(actual, expected, abs_tol=1e-9)
            for actual, expected in zip(values, expected_axis, strict=True)
        )


def test_controller_configuration_claims_each_command_interface_once():
    controllers = _load_yaml(CONTROL_PACKAGE / "config" / "controllers.yaml")
    manager = controllers["controller_manager"]["ros__parameters"]
    assert manager["enforce_command_limits"] is True
    assert manager["omni_base_controller"]["type"] == (
        "omni_wheel_drive_controller/OmniWheelDriveController"
    )

    base = controllers["omni_base_controller"]["ros__parameters"]
    assert base["wheel_names"] == ["wheel_2", "wheel_3", "wheel_1"]
    assert set(base["wheel_names"]) == set(WHEEL_IDS)
    assert base["position_feedback"] is False
    assert base["cmd_vel_timeout"] <= MAX_COMMAND_TIMEOUT_SECONDS
    assert math.isclose(base["wheel_radius"], WHEEL_RADIUS)

    wheel_mount_radii = []
    for wheel_name in WHEEL_IDS:
        mount = _urdf_joints()[f"{wheel_name}_motor_mount"]
        origin = mount.find("origin")
        assert origin is not None
        x, y, _ = (float(value) for value in origin.attrib["xyz"].split())
        wheel_mount_radii.append(math.hypot(x, y))
    assert all(
        math.isclose(base["robot_radius"], radius, abs_tol=1e-6) for radius in wheel_mount_radii
    )

    base_mount_origin = _urdf_joints()["base_mount"].find("origin")
    assert base_mount_origin is not None
    _, _, base_ground_offset = (float(value) for value in base_mount_origin.attrib["xyz"].split())
    assert math.isclose(base_ground_offset, BASE_GROUND_OFFSET)

    arm = controllers["arm_controller"]["ros__parameters"]
    gripper = controllers["gripper_controller"]["ros__parameters"]
    assert arm["joints"] == JOINT_IDS
    assert gripper["joints"] == [GRIPPER_ID]
    assert arm["command_interfaces"] == ["position"]
    assert gripper["command_interfaces"] == ["position"]


def test_moveit_groups_and_controller_actions_match_control_configuration():
    moveit_urdf = (MOVEIT_PACKAGE / "config" / "triskel.urdf.xacro").read_text()
    assert "triskel_description)/urdf/triskel.urdf" in moveit_urdf
    assert "triskel_collision" not in moveit_urdf

    srdf = ET.parse(MOVEIT_PACKAGE / "config" / "triskel.srdf").getroot()
    assert srdf.find("virtual_joint") is None
    groups = {
        group.attrib["name"]: [joint.attrib["name"] for joint in group.findall("joint")]
        for group in srdf.findall("group")
    }
    assert groups == {"arm": JOINT_IDS, "gripper": [GRIPPER_ID]}

    states = {
        state.attrib["name"]: {
            joint.attrib["name"]: float(joint.attrib["value"]) for joint in state.findall("joint")
        }
        for state in srdf.findall("group_state")
    }
    assert [states["home"][joint] for joint in JOINT_IDS] == [
        0.0,
        0.75,
        -0.5,
        0.0,
        0.0,
        -1.0,
        0.0,
    ]
    assert [states["rest"][joint] for joint in JOINT_IDS] == [
        0.0,
        1.6,
        0.1,
        -1.65,
        0.0,
        -0.21,
        0.0,
    ]
    assert states["open"][GRIPPER_ID] == GRIPPER_OPEN_POSITION
    assert states["closed"][GRIPPER_ID] == GRIPPER_CLOSED_POSITION

    moveit_controllers = _load_yaml(MOVEIT_PACKAGE / "config" / "moveit_controllers.yaml")
    simple = moveit_controllers["moveit_simple_controller_manager"]
    assert simple["arm_controller"]["joints"] == JOINT_IDS
    assert simple["gripper_controller"]["joints"] == [GRIPPER_ID]
    assert simple["arm_controller"]["type"] == "FollowJointTrajectory"
    assert simple["gripper_controller"]["type"] == "FollowJointTrajectory"


def test_operator_uses_standard_ros_control_and_recording_interfaces():
    servo = _load_yaml(MOVEIT_PACKAGE / "config" / "servo.yaml")
    assert servo["move_group_name"] == "arm"
    assert servo["planning_frame"] == "base_link"
    assert servo["ee_frame_name"] == "gripper_base_link"
    assert servo["robot_link_command_frame"] == "base_link"
    assert servo["command_out_topic"] == "/arm_controller/joint_trajectory"
    assert servo["cartesian_command_in_topic"] == "~/delta_twist_cmds"
    assert servo["publish_joint_positions"] is True
    assert servo["publish_joint_velocities"] is False
    assert servo["check_collisions"] is True
    assert servo["lower_singularity_threshold"] < servo["hard_stop_singularity_threshold"]
    assert servo["lower_singularity_threshold"] > CALIBRATED_HOME_CONDITION

    node = (OPERATOR_PACKAGE / "triskel_operator" / "node.py").read_text()
    assert '"ros2",\n                "bag",\n                "record"' in node
    assert '"ros2",\n                "bag",\n                "play"' in node
    assert "str(recording.bag_path),\n                *RECORD_TOPICS" in node
    replay_invocation = (
        "str(recording.bag_path),\n"
        '                "--disable-keyboard-controls",\n'
        '                "--topics",\n'
        "                *REPLAY_TOPICS"
    )
    assert replay_invocation in node
    replay_topics = node.split("REPLAY_TOPICS =", maxsplit=1)[1].split(")", maxsplit=1)[0]
    assert "TOOL_COMMAND_TOPIC" not in replay_topics
    assert "ARM_COMMAND_TOPIC" in replay_topics
    assert "_servo_paused_for_homing" in node
    assert '"teleop_devices": ["keyboard", "meta_quest"]' in node
    assert '"topic_rates": self._topic_rate_snapshot(now)' in node
    assert "TOPIC_RATE_SPECS" in node
    assert "_require_hardware_acknowledgement(payload)" in node
    assert 'ROBOT = "triskel"' in node
    assert 'states["home"][GRIPPER_JOINT] = gripper_states["open"]' in node
    assert 'states["rest"][GRIPPER_JOINT] = gripper_states["closed"]' in node
    assert 'self._named_states["open"][GRIPPER_JOINT]' in node
    assert 'self._named_states["closed"][GRIPPER_JOINT]' in node

    dashboard = (OPERATOR_PACKAGE / "static" / "index.html").read_text()
    for capability in (
        "Live Triskel model",
        "Keyboard &amp; Meta Quest teleoperation",
        "Meta Quest",
        "Mode &amp; homing",
        "Record &amp; replay",
        "Live topic rates",
    ):
        assert capability in dashboard
    assert "Meta Quest /" not in dashboard
    assert re.findall(r'<p class="eyebrow">(\d{2}) · ([^<]+)</p>', dashboard) == [
        ("01", "Browser visualization"),
        ("02", "Robot state"),
        ("03", "Feedback"),
        ("04", "rosbag2"),
        ("05", "ROS graph"),
        ("06", "Dead-man control"),
    ]
    assert dashboard.index("Record &amp; replay") < dashboard.index("Live topic rates")

    dashboard_styles = (OPERATOR_PACKAGE / "static" / "styles.css").read_text()
    assert "rotate(var(--angle)) translateY(-11px) rotate(-45deg)" in dashboard_styles
    assert re.findall(r"\.mark i:nth-child\(\d\) \{ --angle: (\d+)deg; \}", dashboard_styles) == [
        "0",
        "120",
        "240",
    ]

    dashboard_javascript = (OPERATOR_PACKAGE / "static" / "app.js").read_text()
    assert 'if (snapshot.mode !== "keyboard") state.held.clear();' in dashboard_javascript
    inactive_branch = dashboard_javascript.split("if (!active) {", maxsplit=1)[1].split(
        "}", maxsplit=1
    )[0]
    assert "state.held.delete(command);" in inactive_branch


def test_meta_quest_bridge_owns_reader_connection_and_input_mapping():
    quest_bridge = (OPERATOR_PACKAGE / "triskel_operator" / "quest_bridge.py").read_text()
    assert 'transformations["r"]' in quest_bridge
    assert 'buttons.get("leftJS")' in quest_bridge
    assert 'buttons.get("rightJS")' in quest_bridge
    assert 'buttons.get("A", False)' in quest_bridge
    assert 'buttons.get("Y", False)' in quest_bridge
    assert "MetaQuestBridge(_load_meta_quest_reader)" in quest_bridge
    assert "sample_token == self._last_sample_token" in quest_bridge
    assert '["adb", "connect", target]' in quest_bridge
    assert "now = time.monotonic()\n            self._last_sample_at = now" in quest_bridge
    assert "QuestReaderSession(reader)" in quest_bridge
    assert "from triskel_operator.topics import QUEST_JOY_TOPIC, QUEST_POSE_TOPIC" in quest_bridge


def test_operator_topic_contract_has_one_internal_source():
    node = (OPERATOR_PACKAGE / "triskel_operator" / "node.py").read_text()
    quest_bridge = (OPERATOR_PACKAGE / "triskel_operator" / "quest_bridge.py").read_text()
    topics = (OPERATOR_PACKAGE / "triskel_operator" / "topics.py").read_text()

    assert "from triskel_operator.topics import (" in node
    assert 'BASE_COMMAND_TOPIC = "/cmd_vel"' in topics
    assert 'ARM_COMMAND_TOPIC = "/arm_controller/joint_trajectory"' in topics
    assert 'GRIPPER_COMMAND_TOPIC = "/gripper_controller/joint_trajectory"' in topics
    assert f'QUEST_POSE_TOPIC = "{DEFAULT_QUEST_POSE_TOPIC}"' in topics
    assert f'QUEST_JOY_TOPIC = "{DEFAULT_QUEST_JOY_TOPIC}"' in topics
    for consumer in (node, quest_bridge):
        assert DEFAULT_QUEST_POSE_TOPIC not in consumer
        assert DEFAULT_QUEST_JOY_TOPIC not in consumer
    assert "GRIPPER_OPEN_POSITION" not in node
    assert "GRIPPER_CLOSED_POSITION" not in node


def test_browser_visualization_uses_canonical_ros_feedback(tmp_path):
    model_module_path = VISUALIZATION_PACKAGE / "triskel_visualization" / "model.py"
    namespace: dict[str, object] = {}
    exec(compile(model_module_path.read_text(), str(model_module_path), "exec"), namespace)
    write_resolved_urdf = cast(
        Callable[[Path, Path, Path], Path],
        namespace["write_resolved_urdf"],
    )
    resolved_path = write_resolved_urdf(
        URDF_PATH,
        DESCRIPTION_PACKAGE,
        tmp_path / "triskel.urdf",
    )
    resolved_root = ET.parse(resolved_path).getroot()
    for mesh in resolved_root.iter("mesh"):
        mesh_path = Path(mesh.attrib["filename"])
        assert mesh_path.is_absolute()
        assert mesh_path.is_file()

    symlinked_share = tmp_path / "installed_share"
    symlinked_meshes = symlinked_share / "meshes"
    symlinked_urdf = symlinked_share / "urdf"
    symlinked_meshes.mkdir(parents=True)
    symlinked_urdf.mkdir()
    for mesh in (DESCRIPTION_PACKAGE / "meshes").iterdir():
        (symlinked_meshes / mesh.name).symlink_to(mesh)
    (symlinked_urdf / "triskel.urdf").symlink_to(URDF_PATH)
    symlink_resolved_path = write_resolved_urdf(
        symlinked_urdf / "triskel.urdf",
        symlinked_share,
        tmp_path / "symlink-install.urdf",
    )
    for mesh in ET.parse(symlink_resolved_path).getroot().iter("mesh"):
        assert Path(mesh.attrib["filename"]).is_file()

    node = (VISUALIZATION_PACKAGE / "triskel_visualization" / "node.py").read_text()
    assert 'JOINT_STATE_TOPIC = "/joint_states"' in node
    assert 'ODOMETRY_TOPIC = "/omni_base_controller/odom"' in node
    assert 'READY_TOPIC = "/triskel/visualization/ready"' in node
    assert "ViserUrdf" in node
    assert "load_collision_meshes=False" in node
    assert "world_axes.visible = False" in node
    assert 'ARM_COMMAND_TOPIC = "/arm_controller/joint_trajectory"' in node
    assert 'GRIPPER_COMMAND_TOPIC = "/gripper_controller/joint_trajectory"' in node
    assert 'TOOL_COMMAND_TOPIC = "/servo_node/delta_twist_cmds"' in node
    assert 'root_node_name="/triskel_base/measured"' in node
    assert 'root_node_name="/triskel_base/commanded"' in node
    assert "mesh_color_override=COMMAND_GHOST_COLOR" in node
    assert "TaskSpaceGripper" in node
    assert "add_frame(root_node_name, show_axes=False)" in node
    assert "TOOL_COMMAND_TIMEOUT_SECONDS = 0.3" in node
    assert "def _reset_task_target_from_measured" in node
    assert "command_stream_stale" in node
    assert 'add_frame("/triskel_base", show_axes=False)' in node
    assert '"/triskel_base",\n            visible=False,' not in node

    dashboard = (OPERATOR_PACKAGE / "static" / "index.html").read_text()
    dashboard_javascript = (OPERATOR_PACKAGE / "static" / "app.js").read_text()
    assert 'id="visualization-frame"' in dashboard
    assert 'class="visualization-legend"' in dashboard
    assert 'id="topic-rate-grid"' in dashboard
    assert "snapshot.visualization" in dashboard_javascript
    assert "visualization.port" in dashboard_javascript
    assert "renderTopicRates(snapshot.topic_rates)" in dashboard_javascript


def test_visualization_dependency_versions_have_one_source():
    setup = (VISUALIZATION_PACKAGE / "setup.py").read_text()
    requirements = (VISUALIZATION_PACKAGE / "requirements.txt").read_text().splitlines()

    assert requirements == ["viser==1.1.0", "yourdfpy==0.0.60"]
    assert 'PACKAGE_DIRECTORY / "requirements.txt"' in setup
    assert all(requirement not in setup for requirement in requirements)


def test_real_hardware_is_opt_in_and_dependency_is_pinned():
    bringup_xacro = (ROS_SRC / "triskel_bringup" / "urdf" / "triskel.urdf.xacro").read_text()
    assert '<xacro:arg name="use_mock_hardware" default="true"' in bringup_xacro

    repos = _load_yaml(REPO_ROOT / "triskel.repos")["repositories"]
    driver = repos["sts_hardware_interface"]
    assert driver["type"] == "git"
    assert driver["url"].startswith("https://github.com/")
    assert COMMIT_PATTERN.fullmatch(driver["version"])

    validation = _load_yaml(HARDWARE_PACKAGE / "config" / "hardware_validation.yaml")
    required_checks = validation["hardware_validation"]["required_checks"]
    assert "gripper_direction_and_limits_verified" in required_checks
    assert "emergency_stop_disables_all_torque" in required_checks
    assert "no_motion_on_controller_activation" in required_checks


def test_ros_launch_files_are_valid_python():
    launch_files = sorted(ROS_SRC.glob("*/launch/*.launch.py"))
    assert launch_files
    for launch_file in launch_files:
        compile(launch_file.read_text(), str(launch_file), "exec")

    operator_launch = (ROS_SRC / "triskel_bringup" / "launch" / "operator.launch.py").read_text()
    assert '("servo_node_main", "servo_node")' in operator_launch
    assert 'package="triskel_visualization"' in operator_launch
    assert 'executable="meta_quest_bridge"' in operator_launch
    assert 'DeclareLaunchArgument("start_visualizer", default_value="true")' in operator_launch
    assert (
        'DeclareLaunchArgument("start_meta_quest_bridge", default_value="true")' in operator_launch
    )
    assert 'DeclareLaunchArgument("quest_ip", default_value="auto")' in operator_launch


def test_docker_smoke_suite_is_modular():
    smoke_runner, smoke_modules, _ = _docker_smoke_sources()
    assert {path.name for path in smoke_modules} == {
        "helpers.bash",
        "test_description.bash",
        "test_homing.bash",
        "test_keyboard_teleop.bash",
        "test_meta_quest.bash",
        "test_recording.bash",
        "test_runtime.bash",
    }
    assert len(smoke_runner.splitlines()) < MAX_SMOKE_RUNNER_LINES
    assert 'source "${script_directory}/tests/helpers.bash"' in smoke_runner
    assert "tests/test_*.bash" in smoke_runner


def test_docker_image_contains_pinned_meta_quest_runtime():
    dockerfile = (DOCKER_DIRECTORY / "Dockerfile.ros2").read_text()
    quest_requirements = (OPERATOR_PACKAGE / "requirements.txt").read_text()

    assert "triskel_operator/requirements.txt" in dockerfile
    assert "git-lfs" in dockerfile
    assert "adb" in dockerfile
    assert "oculus-reader @ git+https://github.com/jborbik/oculus_reader.git@" in quest_requirements


def test_docker_smoke_environment_is_mock_first_and_covers_runtime_contracts():
    dockerfile = (DOCKER_DIRECTORY / "Dockerfile.ros2").read_text()
    compose = _load_yaml(DOCKER_DIRECTORY / "compose.ros2.yaml")
    _, _, smoke_test = _docker_smoke_sources()

    assert "FROM ros:${ROS_DISTRO}-ros-base-noble" in dockerfile
    assert "vcs import" in dockerfile
    assert "rosdep install" in dockerfile
    assert "colcon build" in dockerfile
    assert "triskel_visualization/requirements.txt" in dockerfile
    assert "PYTHONPATH=/opt/triskel-python" in dockerfile

    test_service = compose["services"]["ros2-test"]
    assert compose["name"] == "triskel-ros2"
    assert test_service["image"] == "triskel-ros2-jazzy:latest"
    assert test_service["build"]["args"]["ROS_DISTRO"] == "jazzy"
    assert test_service["profiles"] == ["test"]
    assert "devices" not in test_service
    assert "privileged" not in test_service

    sim_service = compose["services"]["ros2-sim"]
    assert "use_mock_hardware:=true" in sim_service["command"]
    assert "dashboard_host:=0.0.0.0" in sim_service["command"]
    assert "visualizer_host:=0.0.0.0" in sim_service["command"]
    assert "quest_ip:=${TRISKEL_QUEST_IP:-auto}" in sim_service["command"]
    assert sim_service["ports"] == ["127.0.0.1:8765:8765", "127.0.0.1:8080:8080"]
    assert "/api/status" in sim_service["healthcheck"]["test"][1]
    assert "devices" not in sim_service
    assert "privileged" not in sim_service

    hardware_service = compose["services"]["ros2-hardware"]
    assert hardware_service["profiles"] == ["hardware"]
    assert "use_mock_hardware:=false" in hardware_service["command"]
    assert "serial_port:=/dev/triskel" in hardware_service["command"]
    assert "baud_rate:=${TRISKEL_BAUD_RATE:-1000000}" in hardware_service["command"]
    assert hardware_service["devices"] == ["${TRISKEL_SERIAL_PORT:-/dev/ttyACM0}:/dev/triskel:rw"]
    assert hardware_service["ports"] == sim_service["ports"]
    assert '.status.runtime == "real"' in hardware_service["healthcheck"]["test"][1]
    assert "privileged" not in hardware_service

    assert "use_mock_hardware:=true" in smoke_test
    assert "ros2 launch triskel_bringup operator.launch.py" in smoke_test
    for controller in (
        "joint_state_broadcaster",
        "omni_base_controller",
        "arm_controller",
        "gripper_controller",
    ):
        assert controller in smoke_test
    assert "/joint_states" in smoke_test
    assert "/cmd_vel" in smoke_test
    assert "/omni_base_controller/odom" in smoke_test
    assert "/api/mode" in smoke_test
    assert "/api/recording/start" in smoke_test
    assert "/api/replay/start" in smoke_test
    assert "/triskel/teleop/meta_quest/right_controller_pose" in smoke_test
    assert "/triskel/teleop/meta_quest/joy" in smoke_test
    assert "/triskel/visualization/ready" in smoke_test
    assert "http://127.0.0.1:8080/" in smoke_test
    assert "MoveIt Servo did not move an arm joint" in smoke_test


def test_runtime_services_persist_meta_quest_adb_authorization():
    compose = _load_yaml(DOCKER_DIRECTORY / "compose.ros2.yaml")
    sim_service = compose["services"]["ros2-sim"]
    hardware_service = compose["services"]["ros2-hardware"]

    assert sim_service["volumes"] == [
        "quest-adb:/root/.android",
        "../recordings:/recordings",
    ]
    assert hardware_service["volumes"] == sim_service["volumes"]
    assert "quest-adb" in compose["volumes"]
    assert "recording_root:=/recordings" in sim_service["command"]
    assert "recording_root:=/recordings" in hardware_service["command"]


def test_root_command_wraps_lifecycle_quality_and_smoke_workflows():
    project = (REPO_ROOT / "pyproject.toml").read_text()
    command_path = REPO_ROOT / "triskel"
    command = command_path.read_text()

    assert command_path.stat().st_mode & 0o111
    assert "package = false" in project
    for subcommand in (
        "start",
        "stop",
        "status",
        "logs",
        "dashboard",
        "recordings",
        "check",
        "format",
        "smoke",
    ):
        assert f"{subcommand})" in command
    assert "--detach --wait --wait-timeout" in command
    assert "down --remove-orphans" in command
    assert "--hardware" in command
    assert "--serial-port" in command
    assert "--quest-ip" in command
    assert 'export TRISKEL_QUEST_IP="${quest_ip:-auto}"' in command
    assert "Recordings:    %s/recordings" in command
    assert "TRISKEL_SSH_TARGET" in command
    assert "TRISKEL_REMOTE_ROOT" in command
    assert "-L 8765:127.0.0.1:8765" in command
    assert "-L 8080:127.0.0.1:8080" in command
    assert '"${ssh_target}:${remote_root%/}/recordings/"' in command
    assert 'default_serial_port="/dev/ttyACM0"' in command
    assert "bash -n triskel docker/ros_entrypoint.sh docker/test_ros2.sh" in command
    assert "uv run ruff format --check" in command
    assert "uv run ruff check" in command
    assert "uv run ty check ros_ws/src --no-force-exclude --ignore unresolved-import" in command
    assert "uv run pytest" in command
    assert "compose --profile test run --rm ros2-test" in command
