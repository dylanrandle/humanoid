"""ROS node backing the Triskel operator dashboard."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, TwistStamped
from moveit_msgs.srv import ServoCommandType
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from triskel_operator.catalog import Recording, RecordingCatalog
from triskel_operator.http_server import ApiError, make_server
from triskel_operator.rates import TopicRate
from triskel_operator.topics import (
    ARM_COMMAND_TOPIC,
    BASE_COMMAND_TOPIC,
    GRIPPER_COMMAND_TOPIC,
    JOINT_STATE_TOPIC,
    ODOMETRY_TOPIC,
    QUEST_JOY_TOPIC,
    QUEST_POSE_TOPIC,
    SERVO_STATUS_TOPIC,
    TF_STATIC_TOPIC,
    TF_TOPIC,
    TOOL_COMMAND_TOPIC,
    VISUALIZATION_READY_TOPIC,
)

ROBOT = "triskel"
ARM_JOINTS = tuple(f"arm_{index}" for index in range(1, 8))
GRIPPER_JOINT = "gripper_1"
RECORD_TOPICS = (
    BASE_COMMAND_TOPIC,
    TOOL_COMMAND_TOPIC,
    ARM_COMMAND_TOPIC,
    GRIPPER_COMMAND_TOPIC,
    QUEST_POSE_TOPIC,
    QUEST_JOY_TOPIC,
    JOINT_STATE_TOPIC,
    ODOMETRY_TOPIC,
    SERVO_STATUS_TOPIC,
    TF_TOPIC,
    TF_STATIC_TOPIC,
)
# Replay the controller-level commands that were actually executed. Replaying
# Cartesian inputs as well would make Servo generate a second, competing arm stream.
REPLAY_TOPICS = (
    BASE_COMMAND_TOPIC,
    ARM_COMMAND_TOPIC,
    GRIPPER_COMMAND_TOPIC,
)
VALID_TELEOP_COMMANDS = frozenset(
    {
        "tool_forward",
        "tool_backward",
        "tool_left",
        "tool_right",
        "tool_up",
        "tool_down",
        "tool_roll_left",
        "tool_roll_right",
        "tool_pitch_up",
        "tool_pitch_down",
        "tool_yaw_left",
        "tool_yaw_right",
        "base_forward",
        "base_backward",
        "base_left",
        "base_right",
        "base_yaw_left",
        "base_yaw_right",
        "gripper_open",
        "gripper_close",
    }
)
TELEOP_TIMEOUT_SECONDS = 0.3
JOINT_STATE_TIMEOUT_SECONDS = 2.0
VISUALIZATION_TIMEOUT_SECONDS = 2.5
RECENT_TOPIC_TIMEOUT_SECONDS = 0.5
HOMING_TOLERANCE = 0.03
CONTROL_PERIOD_SECONDS = 0.05
QUATERNION_MIN_NORM = 1e-6
VECTOR_EPSILON = 1e-9
TOPIC_RATE_SPECS = (
    (JOINT_STATE_TOPIC, "Joint feedback", 30.0, "always"),
    (ODOMETRY_TOPIC, "Base odometry", 20.0, "always"),
    (VISUALIZATION_READY_TOPIC, "Visualizer heartbeat", 0.5, "visualization"),
    (BASE_COMMAND_TOPIC, "Base command", 10.0, "base_command"),
    (TOOL_COMMAND_TOPIC, "Task-space command", 10.0, "tool_command"),
    (ARM_COMMAND_TOPIC, "Limited arm command", 5.0, "tool_command"),
    (GRIPPER_COMMAND_TOPIC, "Gripper command", 10.0, "recent"),
    (QUEST_POSE_TOPIC, "Quest controller pose", 5.0, "meta_quest"),
    (QUEST_JOY_TOPIC, "Quest buttons / axes", 5.0, "meta_quest"),
)


class TriskelOperator(Node):
    def __init__(self) -> None:
        super().__init__("triskel_operator")
        self._declare_parameters()
        self._load_parameters()
        self._initialize_state()

        recording_root = str(self.get_parameter("recording_root").value)
        self._catalog = RecordingCatalog(recording_root)
        self._named_states = self._load_named_states()
        self._create_ros_interfaces()
        self._start_http_server()

    def _declare_parameters(self) -> None:
        self.declare_parameter("runtime", "sim")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("recording_root", "~/.ros/triskel/recordings")
        self.declare_parameter("visualization_enabled", True)
        self.declare_parameter("visualization_port", 8080)
        self.declare_parameter("tool_linear_speed", 0.05)
        self.declare_parameter("tool_angular_speed", 0.3)
        self.declare_parameter("base_linear_speed", 0.1)
        self.declare_parameter("base_angular_speed", 0.3)
        self.declare_parameter("gripper_speed", 0.5)
        self.declare_parameter("vr_left_grip_button", 4)
        self.declare_parameter("vr_right_grip_button", 5)
        self.declare_parameter("vr_close_button", 0)
        self.declare_parameter("vr_open_button", 1)
        self.declare_parameter("vr_home_button", 2)
        self.declare_parameter("vr_record_button", 3)
        self.declare_parameter("vr_base_strafe_axis", 0)
        self.declare_parameter("vr_base_forward_axis", 1)
        self.declare_parameter("vr_base_yaw_axis", 3)
        self.declare_parameter("vr_deadzone", 0.1)

    def _load_parameters(self) -> None:
        self.runtime = str(self.get_parameter("runtime").value)
        if self.runtime not in {"sim", "real"}:
            raise ValueError("runtime must be 'sim' or 'real'")
        self._tool_linear_speed = float(self.get_parameter("tool_linear_speed").value)
        self._tool_angular_speed = float(self.get_parameter("tool_angular_speed").value)
        self._base_linear_speed = float(self.get_parameter("base_linear_speed").value)
        self._base_angular_speed = float(self.get_parameter("base_angular_speed").value)
        self._gripper_speed = float(self.get_parameter("gripper_speed").value)
        self._visualization_enabled = bool(self.get_parameter("visualization_enabled").value)
        self._visualization_port = int(self.get_parameter("visualization_port").value)
        self._vr_left_grip_button = int(self.get_parameter("vr_left_grip_button").value)
        self._vr_right_grip_button = int(self.get_parameter("vr_right_grip_button").value)
        self._vr_close_button = int(self.get_parameter("vr_close_button").value)
        self._vr_open_button = int(self.get_parameter("vr_open_button").value)
        self._vr_home_button = int(self.get_parameter("vr_home_button").value)
        self._vr_record_button = int(self.get_parameter("vr_record_button").value)
        self._vr_base_strafe_axis = int(self.get_parameter("vr_base_strafe_axis").value)
        self._vr_base_forward_axis = int(self.get_parameter("vr_base_forward_axis").value)
        self._vr_base_yaw_axis = int(self.get_parameter("vr_base_yaw_axis").value)
        self._vr_deadzone = float(self.get_parameter("vr_deadzone").value)

    def _initialize_state(self) -> None:
        self._lock = threading.RLock()
        self._joint_positions: dict[str, float] = {}
        self._joint_state_at: float | None = None
        self._odom_at: float | None = None
        self._visualization_at: float | None = None
        self._mode = "idle"
        self._preset: str | None = None
        self._last_error: str | None = None
        self._teleop_commands: frozenset[str] = frozenset()
        self._teleop_at = 0.0
        self._teleop_was_active = False
        self._vr_pose: tuple[float, ...] | None = None
        self._vr_pose_at = 0.0
        self._vr_previous_pose: tuple[float, ...] | None = None
        self._vr_previous_pose_at = 0.0
        self._vr_axes: tuple[float, ...] = ()
        self._vr_buttons: tuple[bool, ...] = ()
        self._vr_joy_at = 0.0
        self._vr_was_active = False
        self._vr_home_requested = False
        self._vr_record_toggle_requested = False
        self._gripper_command: float | None = None
        self._homing_target: dict[str, float] | None = None
        self._homing_deadline: float | None = None
        self._homing_duration: float | None = None
        self._homing_pause_request_id = 0
        self._servo_configuring = False
        self._servo_configured = False
        self._record_process: subprocess.Popen[bytes] | None = None
        self._recording: Recording | None = None
        self._replay_process: subprocess.Popen[bytes] | None = None
        self._replay_recording: Recording | None = None
        self._replay_outcome: str | None = None
        self._topic_rates = {topic: TopicRate() for topic, *_ in TOPIC_RATE_SPECS}

    def _create_ros_interfaces(self) -> None:
        self._base_publisher = self.create_publisher(TwistStamped, BASE_COMMAND_TOPIC, 10)
        self._tool_publisher = self.create_publisher(TwistStamped, TOOL_COMMAND_TOPIC, 10)
        self._arm_publisher = self.create_publisher(JointTrajectory, ARM_COMMAND_TOPIC, 10)
        self._gripper_publisher = self.create_publisher(JointTrajectory, GRIPPER_COMMAND_TOPIC, 10)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_state_callback, 10)
        self.create_subscription(Odometry, ODOMETRY_TOPIC, self._odometry_callback, 10)
        self.create_subscription(PoseStamped, QUEST_POSE_TOPIC, self._vr_pose_callback, 10)
        self.create_subscription(Joy, QUEST_JOY_TOPIC, self._vr_joy_callback, 10)
        self.create_subscription(
            Bool,
            VISUALIZATION_READY_TOPIC,
            self._visualization_ready_callback,
            10,
        )
        for message_type, topic in (
            (TwistStamped, BASE_COMMAND_TOPIC),
            (TwistStamped, TOOL_COMMAND_TOPIC),
            (JointTrajectory, ARM_COMMAND_TOPIC),
            (JointTrajectory, GRIPPER_COMMAND_TOPIC),
        ):
            self.create_subscription(
                message_type,
                topic,
                partial(self._topic_rate_callback, topic),
                10,
            )
        self._servo_pause = self.create_client(SetBool, "/servo_node/pause_servo")
        self._servo_switch = self.create_client(ServoCommandType, "/servo_node/switch_command_type")
        self.create_timer(CONTROL_PERIOD_SECONDS, self._control_tick)
        self.create_timer(0.2, self._housekeeping_tick)

    def _start_http_server(self) -> None:
        static_root = Path(get_package_share_directory("triskel_operator")) / "static"
        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        self._http_server = make_server(host, port, static_root, self._api)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="triskel-dashboard-http",
            daemon=True,
        )
        self._http_thread.start()
        self.get_logger().info(f"Triskel dashboard available at http://{host}:{port}")

    def _api(self, path: str, payload: dict[str, Any]) -> object:
        routes = {
            "/api/status": self.snapshot,
            "/api/robot": lambda: self.select_robot(payload),
            "/api/mode": lambda: self.select_mode(payload),
            "/api/teleop": lambda: self.update_teleop(payload),
            "/api/recording/start": self.start_recording,
            "/api/recording/stop": self.stop_recording,
            "/api/replay/start": lambda: self.start_replay(payload),
            "/api/replay/stop": self.stop_replay,
        }
        action = routes.get(path)
        if action is None:
            raise ApiError("Unknown endpoint.", HTTPStatus.NOT_FOUND)
        return action()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._refresh_processes_locked()
            now = time.monotonic()
            joint_age = None if self._joint_state_at is None else now - self._joint_state_at
            odom_age = None if self._odom_at is None else now - self._odom_at
            visualization_age = (
                None if self._visualization_at is None else now - self._visualization_at
            )
            vr_pose_age = None if self._vr_pose is None else now - self._vr_pose_at
            vr_joy_age = None if not self._vr_buttons else now - self._vr_joy_at
            controllers = self._controller_status()
            ready = bool(
                joint_age is not None
                and joint_age < JOINT_STATE_TIMEOUT_SECONDS
                and all(controllers.values())
            )
            recordings = [recording.public() for recording in self._catalog.list()]
            return {
                "robot": ROBOT,
                "robots": [ROBOT],
                "runtime": self.runtime,
                "ready": ready,
                "mode": self._mode,
                "preset": self._preset,
                "teleop_devices": ["keyboard", "meta_quest"],
                "teleop_device": (self._mode if self._mode in {"keyboard", "meta_quest"} else None),
                "controllers": controllers,
                "servo_ready": self._servo_configured,
                "joint_state_age": joint_age,
                "odometry_age": odom_age,
                "visualization": {
                    "enabled": self._visualization_enabled,
                    "ready": bool(
                        self._visualization_enabled
                        and visualization_age is not None
                        and visualization_age <= VISUALIZATION_TIMEOUT_SECONDS
                    ),
                    "port": self._visualization_port,
                },
                "meta_quest": {
                    "ready": bool(
                        vr_pose_age is not None
                        and vr_pose_age <= TELEOP_TIMEOUT_SECONDS
                        and vr_joy_age is not None
                        and vr_joy_age <= TELEOP_TIMEOUT_SECONDS
                    ),
                    "pose_age": vr_pose_age,
                    "joy_age": vr_joy_age,
                    "deadman": self._vr_deadman_pressed(),
                },
                "topic_rates": self._topic_rate_snapshot(now),
                "joints": dict(self._joint_positions),
                "recording": {
                    "running": self._record_process is not None,
                    "id": self._recording.id if self._recording is not None else None,
                },
                "recordings": recordings,
                "replay": {
                    "running": self._replay_process is not None,
                    "id": (
                        self._replay_recording.id if self._replay_recording is not None else None
                    ),
                    "outcome": self._replay_outcome,
                },
                "last_error": self._last_error,
            }

    def _topic_rate_callback(self, topic: str, _message: Any) -> None:
        with self._lock:
            self._topic_rates[topic].observe()

    def _topic_rate_snapshot(self, now: float) -> list[dict[str, object]]:
        keyboard_fresh = (
            self._mode == "keyboard" and now - self._teleop_at <= TELEOP_TIMEOUT_SECONDS
        )
        base_requested = keyboard_fresh and any(
            command.startswith("base_") for command in self._teleop_commands
        )
        tool_requested = keyboard_fresh and any(
            command.startswith("tool_") for command in self._teleop_commands
        )
        quest_motion = self._mode == "meta_quest" and self._vr_deadman_pressed()
        rows: list[dict[str, object]] = []
        for topic, label, minimum_hz, expectation in TOPIC_RATE_SPECS:
            hz, age = self._topic_rates[topic].sample(now)
            expected = {
                "always": True,
                "visualization": self._visualization_enabled,
                "base_command": base_requested or quest_motion,
                "tool_command": tool_requested or quest_motion,
                "meta_quest": self._mode == "meta_quest",
                "recent": age is not None and age <= RECENT_TOPIC_TIMEOUT_SECONDS,
            }[expectation]
            freshness_limit = (
                VISUALIZATION_TIMEOUT_SECONDS
                if expectation == "visualization"
                else max(0.5, 2.5 / minimum_hz)
            )
            if not expected:
                state = "idle"
            elif age is not None and age <= freshness_limit and hz >= minimum_hz:
                state = "healthy"
            else:
                state = "unhealthy"
            rows.append(
                {
                    "topic": topic,
                    "label": label,
                    "hz": round(hz, 1),
                    "minimum_hz": minimum_hz,
                    "age": None if age is None else round(age, 2),
                    "state": state,
                }
            )
        return rows

    def select_robot(self, payload: dict[str, Any]) -> dict[str, object]:
        if payload.get("robot") != ROBOT:
            raise ApiError("Triskel is the only supported robot.")
        return self.snapshot()

    def select_mode(self, payload: dict[str, Any]) -> dict[str, object]:
        requested = payload.get("mode")
        if requested not in {"idle", "home", "rest", "keyboard", "meta_quest"}:
            raise ApiError("Mode must be idle, home, rest, keyboard, or meta_quest.")
        if requested != "idle":
            self._require_ready()
            self._require_hardware_acknowledgement(payload)
        with self._lock:
            if self._replay_process is not None:
                raise ApiError("Stop replay before changing control mode.", HTTPStatus.CONFLICT)
            if requested == "idle":
                self._enter_idle_locked()
            elif requested in {"home", "rest"}:
                self._start_homing_locked(str(requested))
            else:
                if not self._servo_configured:
                    raise ApiError("MoveIt Servo is not ready.", HTTPStatus.CONFLICT)
                self._homing_target = None
                self._homing_deadline = None
                self._mode = str(requested)
                self._preset = None
                self._reset_teleop_locked()
                self._set_servo_paused(False)
        return self.snapshot()

    def update_teleop(self, payload: dict[str, Any]) -> dict[str, object]:
        commands = payload.get("commands")
        if not isinstance(commands, list) or not all(isinstance(item, str) for item in commands):
            raise ApiError("commands must be a list of command names.")
        command_set = frozenset(commands)
        unknown = command_set - VALID_TELEOP_COMMANDS
        if unknown:
            raise ApiError(f"Unknown teleop command: {sorted(unknown)[0]}")
        with self._lock:
            if self._mode != "keyboard":
                raise ApiError(
                    "Select keyboard teleop before sending browser motion.",
                    HTTPStatus.CONFLICT,
                )
            self._teleop_commands = command_set
            self._teleop_at = time.monotonic()
        return self.snapshot()

    def start_recording(self) -> dict[str, object]:
        self._require_ready()
        with self._lock:
            self._refresh_processes_locked()
            if self._record_process is not None:
                raise ApiError("Recording is already active.", HTTPStatus.CONFLICT)
            recording = self._catalog.create(robot=ROBOT, runtime=self.runtime)
            command = [
                "ros2",
                "bag",
                "record",
                "--disable-keyboard-controls",
                "--output",
                str(recording.bag_path),
                *RECORD_TOPICS,
            ]
            self._record_process = self._start_process(command)
            self._recording = recording
        return self.snapshot()

    def stop_recording(self) -> dict[str, object]:
        with self._lock:
            process = self._record_process
        if process is None:
            raise ApiError("Recording is not active.", HTTPStatus.CONFLICT)
        return_code = self._stop_process(process)
        with self._lock:
            self._record_process = None
            if return_code not in {0, -signal.SIGINT}:
                self._last_error = f"rosbag2 recorder exited with code {return_code}."
        return self.snapshot()

    def start_replay(self, payload: dict[str, Any]) -> dict[str, object]:
        self._require_ready()
        self._require_hardware_acknowledgement(payload)
        recording_id = payload.get("recording")
        if not isinstance(recording_id, str):
            raise ApiError("Select a recording to replay.")
        try:
            recording = self._catalog.get(recording_id)
        except ValueError as exc:
            raise ApiError(str(exc)) from exc
        if recording.robot != ROBOT:
            raise ApiError("The selected recording is for another robot.", HTTPStatus.CONFLICT)
        with self._lock:
            self._refresh_processes_locked()
            if self._record_process is not None:
                raise ApiError("Stop recording before replaying.", HTTPStatus.CONFLICT)
            if self._replay_process is not None:
                raise ApiError("Replay is already active.", HTTPStatus.CONFLICT)
            self._enter_idle_locked()
            command = [
                "ros2",
                "bag",
                "play",
                str(recording.bag_path),
                "--disable-keyboard-controls",
                "--topics",
                *REPLAY_TOPICS,
            ]
            self._replay_process = self._start_process(command)
            self._replay_recording = recording
            self._replay_outcome = None
            self._mode = "replay"
        return self.snapshot()

    def stop_replay(self) -> dict[str, object]:
        with self._lock:
            process = self._replay_process
        if process is None:
            raise ApiError("Replay is not active.", HTTPStatus.CONFLICT)
        return_code = self._stop_process(process)
        with self._lock:
            self._replay_process = None
            self._replay_outcome = "stopped" if return_code != 0 else "completed"
            self._enter_idle_locked()
        return self.snapshot()

    def _joint_state_callback(self, message: JointState) -> None:
        with self._lock:
            self._topic_rates[JOINT_STATE_TOPIC].observe()
            self._joint_positions.update(zip(message.name, message.position, strict=False))
            self._joint_state_at = time.monotonic()
            if self._gripper_command is None and GRIPPER_JOINT in self._joint_positions:
                self._gripper_command = self._joint_positions[GRIPPER_JOINT]

    def _odometry_callback(self, _message: Odometry) -> None:
        with self._lock:
            self._topic_rates[ODOMETRY_TOPIC].observe()
            self._odom_at = time.monotonic()

    def _visualization_ready_callback(self, message: Bool) -> None:
        with self._lock:
            self._topic_rates[VISUALIZATION_READY_TOPIC].observe()
            self._visualization_at = time.monotonic() if message.data else None

    def _vr_pose_callback(self, message: PoseStamped) -> None:
        pose = message.pose
        sample = (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        if not all(math.isfinite(value) for value in sample):
            return
        quaternion_norm = math.sqrt(sum(value * value for value in sample[3:]))
        if quaternion_norm < QUATERNION_MIN_NORM:
            return
        normalized = (*sample[:3], *(value / quaternion_norm for value in sample[3:]))
        with self._lock:
            self._topic_rates[QUEST_POSE_TOPIC].observe()
            self._vr_pose = normalized
            self._vr_pose_at = time.monotonic()

    def _vr_joy_callback(self, message: Joy) -> None:
        axes = tuple(float(value) if math.isfinite(value) else 0.0 for value in message.axes)
        buttons = tuple(bool(value) for value in message.buttons)
        with self._lock:
            self._topic_rates[QUEST_JOY_TOPIC].observe()
            previous = self._vr_buttons
            self._vr_axes = axes
            self._vr_buttons = buttons
            self._vr_joy_at = time.monotonic()
            if self._mode == "meta_quest":
                if self._button(buttons, self._vr_home_button) and not self._button(
                    previous, self._vr_home_button
                ):
                    self._vr_home_requested = True
                if self._button(buttons, self._vr_record_button) and not self._button(
                    previous, self._vr_record_button
                ):
                    self._vr_record_toggle_requested = True

    def _control_tick(self) -> None:
        with self._lock:
            if self._mode == "meta_quest":
                self._publish_vr_teleop_locked()
                return
            if self._mode != "keyboard":
                return
            commands = self._teleop_commands
            fresh = time.monotonic() - self._teleop_at <= TELEOP_TIMEOUT_SECONDS
            if not fresh:
                commands = frozenset()
                self._teleop_commands = commands
            if not commands:
                if self._teleop_was_active:
                    self._publish_zero_motion()
                    self._teleop_was_active = False
                return
            self._publish_teleop(commands)
            self._teleop_was_active = True

    def _housekeeping_tick(self) -> None:
        home_requested = False
        record_toggle_requested = False
        with self._lock:
            if not self._servo_configured:
                self._configure_servo()
            self._refresh_processes_locked()
            self._check_homing_locked()
            if self._mode == "meta_quest":
                home_requested = self._vr_home_requested
                record_toggle_requested = self._vr_record_toggle_requested
            self._vr_home_requested = False
            self._vr_record_toggle_requested = False
        if home_requested:
            with self._lock:
                if self._mode == "meta_quest":
                    try:
                        self._start_homing_locked("home", preserve_gripper=True)
                    except ApiError as exc:
                        self._last_error = str(exc)
        if record_toggle_requested:
            try:
                if self._record_process is None:
                    self.start_recording()
                else:
                    self.stop_recording()
            except ApiError as exc:
                with self._lock:
                    self._last_error = str(exc)

    def _configure_servo(self) -> None:
        if self._servo_configuring or not (
            self._servo_pause.service_is_ready() and self._servo_switch.service_is_ready()
        ):
            return
        self._servo_configuring = True
        pause_request = SetBool.Request()
        pause_request.data = True
        future = self._servo_pause.call_async(pause_request)
        future.add_done_callback(self._servo_paused_for_configuration)

    def _servo_paused_for_configuration(self, future: Any) -> None:
        try:
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError("MoveIt Servo rejected the pause request.")
            switch_request = ServoCommandType.Request()
            switch_request.command_type = ServoCommandType.Request.TWIST
            switch_future = self._servo_switch.call_async(switch_request)
            switch_future.add_done_callback(self._servo_command_type_configured)
        except Exception as exc:
            with self._lock:
                self._last_error = f"MoveIt Servo configuration failed: {exc}"
                self._servo_configuring = False

    def _servo_command_type_configured(self, future: Any) -> None:
        try:
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError("MoveIt Servo rejected Twist command mode.")
        except Exception as exc:
            with self._lock:
                self._last_error = f"MoveIt Servo configuration failed: {exc}"
                self._servo_configuring = False
            return
        with self._lock:
            self._servo_configured = True
            self._servo_configuring = False
            self._last_error = None

    def _set_servo_paused(self, paused: bool) -> None:
        if not self._servo_pause.service_is_ready():
            return
        request = SetBool.Request()
        request.data = paused
        self._servo_pause.call_async(request)

    def _publish_teleop(self, commands: frozenset[str]) -> None:
        tool = TwistStamped()
        tool.header.stamp = self.get_clock().now().to_msg()
        tool.header.frame_id = "base_link"
        tool.twist.linear.x = (
            self._axis(commands, "tool_forward", "tool_backward") * self._tool_linear_speed
        )
        tool.twist.linear.y = (
            self._axis(commands, "tool_left", "tool_right") * self._tool_linear_speed
        )
        tool.twist.linear.z = self._axis(commands, "tool_up", "tool_down") * self._tool_linear_speed
        tool.twist.angular.x = (
            self._axis(commands, "tool_roll_left", "tool_roll_right") * self._tool_angular_speed
        )
        tool.twist.angular.y = (
            self._axis(commands, "tool_pitch_up", "tool_pitch_down") * self._tool_angular_speed
        )
        tool.twist.angular.z = (
            self._axis(commands, "tool_yaw_left", "tool_yaw_right") * self._tool_angular_speed
        )
        self._tool_publisher.publish(tool)

        base = TwistStamped()
        base.header.stamp = tool.header.stamp
        base.header.frame_id = "base_link"
        base.twist.linear.x = (
            self._axis(commands, "base_forward", "base_backward") * self._base_linear_speed
        )
        base.twist.linear.y = (
            self._axis(commands, "base_left", "base_right") * self._base_linear_speed
        )
        base.twist.angular.z = (
            self._axis(commands, "base_yaw_left", "base_yaw_right") * self._base_angular_speed
        )
        self._base_publisher.publish(base)

        gripper_direction = self._axis(commands, "gripper_open", "gripper_close")
        self._publish_gripper_direction(gripper_direction)

    def _publish_vr_teleop_locked(self) -> None:
        now = time.monotonic()
        fresh = bool(
            self._vr_pose is not None
            and now - self._vr_pose_at <= TELEOP_TIMEOUT_SECONDS
            and now - self._vr_joy_at <= TELEOP_TIMEOUT_SECONDS
        )
        active = fresh and self._vr_deadman_pressed()
        if not active:
            self._vr_previous_pose = None
            if self._vr_was_active:
                self._publish_zero_motion()
            self._vr_was_active = False
            return

        assert self._vr_pose is not None
        stamp = self.get_clock().now().to_msg()
        tool = TwistStamped()
        tool.header.stamp = stamp
        tool.header.frame_id = "base_link"
        if self._vr_previous_pose is not None and self._vr_pose_at > self._vr_previous_pose_at:
            delta_time = self._vr_pose_at - self._vr_previous_pose_at
            linear, angular = self._vr_pose_delta(
                self._vr_previous_pose,
                self._vr_pose,
                delta_time,
            )
            tool.twist.linear.x, tool.twist.linear.y, tool.twist.linear.z = linear
            tool.twist.angular.x, tool.twist.angular.y, tool.twist.angular.z = angular
        self._tool_publisher.publish(tool)
        self._vr_previous_pose = self._vr_pose
        self._vr_previous_pose_at = self._vr_pose_at

        base = TwistStamped()
        base.header.stamp = stamp
        base.header.frame_id = "base_link"
        base.twist.linear.x = (
            self._deadzone_axis(self._vr_base_forward_axis) * self._base_linear_speed
        )
        base.twist.linear.y = (
            self._deadzone_axis(self._vr_base_strafe_axis) * self._base_linear_speed
        )
        base.twist.angular.z = (
            -self._deadzone_axis(self._vr_base_yaw_axis) * self._base_angular_speed
        )
        self._base_publisher.publish(base)

        gripper_direction = float(self._button(self._vr_buttons, self._vr_open_button)) - float(
            self._button(self._vr_buttons, self._vr_close_button)
        )
        self._publish_gripper_direction(gripper_direction)
        self._vr_was_active = True

    def _vr_pose_delta(
        self,
        previous: tuple[float, ...],
        current: tuple[float, ...],
        delta_time: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        controller_linear = tuple(
            (current[index] - previous[index]) / delta_time for index in range(3)
        )
        linear = self._map_vr_vector(controller_linear)
        linear = self._limit_vector(linear, self._tool_linear_speed)

        previous_quaternion = previous[3:]
        current_quaternion = current[3:]
        delta_quaternion = self._quaternion_multiply(
            current_quaternion,
            self._quaternion_conjugate(previous_quaternion),
        )
        if delta_quaternion[3] < 0.0:
            delta_quaternion = tuple(-value for value in delta_quaternion)
        vector_norm = math.sqrt(sum(value * value for value in delta_quaternion[:3]))
        if vector_norm < VECTOR_EPSILON:
            angular = (0.0, 0.0, 0.0)
        else:
            angle = 2.0 * math.atan2(vector_norm, max(delta_quaternion[3], 0.0))
            controller_angular = tuple(
                value / vector_norm * angle / delta_time for value in delta_quaternion[:3]
            )
            angular = self._limit_vector(
                self._map_vr_vector(controller_angular),
                self._tool_angular_speed,
            )
        return linear, angular

    @staticmethod
    def _map_vr_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
        """Map OpenXR right/up/back coordinates into ROS base-link coordinates."""

        x, y, z = vector
        return x, -z, y

    @staticmethod
    def _limit_vector(
        vector: tuple[float, float, float], maximum: float
    ) -> tuple[float, float, float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= maximum or norm < VECTOR_EPSILON:
            return vector
        scale = maximum / norm
        return tuple(value * scale for value in vector)

    @staticmethod
    def _quaternion_conjugate(
        quaternion: tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        x, y, z, w = quaternion
        return -x, -y, -z, w

    @staticmethod
    def _quaternion_multiply(
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        return (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )

    def _deadzone_axis(self, index: int) -> float:
        value = self._axis_value(self._vr_axes, index)
        return 0.0 if abs(value) < self._vr_deadzone else value

    def _vr_deadman_pressed(self) -> bool:
        return self._button(self._vr_buttons, self._vr_left_grip_button) or self._button(
            self._vr_buttons, self._vr_right_grip_button
        )

    @staticmethod
    def _button(buttons: tuple[bool, ...], index: int) -> bool:
        return 0 <= index < len(buttons) and buttons[index]

    @staticmethod
    def _axis_value(axes: tuple[float, ...], index: int) -> float:
        return axes[index] if 0 <= index < len(axes) else 0.0

    def _publish_gripper_direction(self, direction: float) -> None:
        if not direction or self._gripper_command is None:
            return
        open_position = self._named_states["rest"][GRIPPER_JOINT]
        closed_position = self._named_states["home"][GRIPPER_JOINT]
        lower, upper = sorted((open_position, closed_position))
        self._gripper_command = min(
            upper,
            max(
                lower,
                self._gripper_command - direction * self._gripper_speed * CONTROL_PERIOD_SECONDS,
            ),
        )
        self._publish_trajectory(
            self._gripper_publisher,
            (GRIPPER_JOINT,),
            (self._gripper_command,),
            0.15,
        )

    def _publish_zero_motion(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for publisher, frame in (
            (self._base_publisher, "base_link"),
            (self._tool_publisher, "base_link"),
        ):
            message = TwistStamped()
            message.header.stamp = stamp
            message.header.frame_id = frame
            publisher.publish(message)

    def _start_homing_locked(self, preset: str, *, preserve_gripper: bool = False) -> None:
        if not self._servo_pause.service_is_ready():
            raise ApiError(
                "MoveIt Servo pause service is unavailable.",
                HTTPStatus.CONFLICT,
            )
        target = dict(self._named_states[preset])
        joints = (*ARM_JOINTS, GRIPPER_JOINT)
        missing = [joint for joint in joints if joint not in self._joint_positions]
        if missing:
            raise ApiError("Arm or gripper joint state is incomplete.", HTTPStatus.CONFLICT)
        if preserve_gripper:
            target[GRIPPER_JOINT] = self._joint_positions[GRIPPER_JOINT]
        max_displacement = max(
            abs(target[joint] - self._joint_positions[joint]) for joint in joints
        )
        duration = max(0.5, max_displacement / 0.8)
        self._reset_teleop_locked()
        self._publish_zero_motion()
        self._mode = "homing"
        self._preset = preset
        self._homing_target = target
        self._homing_deadline = None
        self._homing_duration = duration
        self._homing_pause_request_id += 1
        request_id = self._homing_pause_request_id

        request = SetBool.Request()
        request.data = True
        future = self._servo_pause.call_async(request)
        future.add_done_callback(
            lambda completed, expected=request_id: self._servo_paused_for_homing(
                completed, expected
            )
        )

    def _servo_paused_for_homing(self, future: Any, request_id: int) -> None:
        error: str | None = None
        try:
            response = future.result()
            if response is None or not response.success:
                error = "MoveIt Servo rejected the pause request."
        except Exception as exc:
            error = f"MoveIt Servo pause request failed: {exc}"

        with self._lock:
            if (
                request_id != self._homing_pause_request_id
                or self._mode != "homing"
                or self._homing_target is None
                or self._homing_duration is None
            ):
                return
            if error is not None:
                self._last_error = error
                self._mode = "idle"
                self._preset = None
                self._homing_target = None
                self._homing_deadline = None
                self._homing_duration = None
                return

            target = self._homing_target
            duration = self._homing_duration
            self._publish_trajectory(
                self._arm_publisher,
                ARM_JOINTS,
                tuple(target[joint] for joint in ARM_JOINTS),
                duration,
            )
            self._publish_trajectory(
                self._gripper_publisher,
                (GRIPPER_JOINT,),
                (target[GRIPPER_JOINT],),
                duration,
            )
            self._gripper_command = target[GRIPPER_JOINT]
            self._homing_deadline = time.monotonic() + duration + 2.0

    def _check_homing_locked(self) -> None:
        if self._homing_target is None or self._homing_deadline is None:
            return
        if all(joint in self._joint_positions for joint in self._homing_target):
            error = max(
                abs(self._homing_target[joint] - self._joint_positions[joint])
                for joint in self._homing_target
            )
            if error < HOMING_TOLERANCE:
                self._homing_target = None
                self._homing_deadline = None
                self._homing_duration = None
                self._mode = "idle"
                self._preset = None
                return
        if time.monotonic() > self._homing_deadline:
            self._last_error = f"{self._preset or 'Homing'} did not reach its target in time."
            self._homing_target = None
            self._homing_deadline = None
            self._homing_duration = None
            self._mode = "idle"
            self._preset = None

    def _enter_idle_locked(self) -> None:
        self._mode = "idle"
        self._preset = None
        self._homing_target = None
        self._homing_deadline = None
        self._homing_duration = None
        self._homing_pause_request_id += 1
        self._reset_teleop_locked()
        self._publish_zero_motion()
        self._set_servo_paused(True)

    def _reset_teleop_locked(self) -> None:
        self._teleop_commands = frozenset()
        self._teleop_was_active = False
        self._vr_previous_pose = None
        self._vr_previous_pose_at = 0.0
        self._vr_was_active = False

    @staticmethod
    def _publish_trajectory(
        publisher: Any,
        joints: tuple[str, ...],
        positions: tuple[float, ...],
        seconds: float,
    ) -> None:
        whole_seconds = math.floor(seconds)
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = Duration(
            sec=whole_seconds,
            nanosec=int((seconds - whole_seconds) * 1_000_000_000),
        )
        trajectory = JointTrajectory()
        trajectory.joint_names = list(joints)
        trajectory.points = [point]
        publisher.publish(trajectory)

    @staticmethod
    def _axis(commands: frozenset[str], positive: str, negative: str) -> float:
        return float(positive in commands) - float(negative in commands)

    def _require_ready(self) -> None:
        status = self.snapshot()
        if not status["ready"]:
            raise ApiError(
                "Wait for all controllers and joint state feedback.", HTTPStatus.CONFLICT
            )

    def _require_hardware_acknowledgement(self, payload: dict[str, Any]) -> None:
        if self.runtime == "real" and payload.get("acknowledge_real_hardware") is not True:
            raise ApiError(
                "Real hardware motion requires an explicit safety acknowledgement.",
                HTTPStatus.CONFLICT,
            )

    def _controller_status(self) -> dict[str, bool]:
        return {
            "base": self.count_subscribers(BASE_COMMAND_TOPIC) > 0,
            "arm": self.count_subscribers(ARM_COMMAND_TOPIC) > 0,
            "gripper": self.count_subscribers(GRIPPER_COMMAND_TOPIC) > 0,
        }

    def _load_named_states(self) -> dict[str, dict[str, float]]:
        path = (
            Path(get_package_share_directory("triskel_moveit_config")) / "config" / "triskel.srdf"
        )
        root = ET.parse(path).getroot()
        states: dict[str, dict[str, float]] = {}
        gripper_states: dict[str, float] = {}
        for state in root.findall("group_state"):
            name = state.attrib.get("name")
            group = state.attrib.get("group")
            if group == "arm" and name in {"home", "rest"}:
                states[name] = {
                    joint.attrib["name"]: float(joint.attrib["value"])
                    for joint in state.findall("joint")
                }
            elif group == "gripper" and name in {"open", "closed"}:
                joint = state.find(f"joint[@name='{GRIPPER_JOINT}']")
                if joint is not None:
                    gripper_states[name] = float(joint.attrib["value"])
        if set(states) != {"home", "rest"}:
            raise RuntimeError("Triskel SRDF must define arm home and rest states.")
        if set(gripper_states) != {"open", "closed"}:
            raise RuntimeError("Triskel SRDF must define open and closed gripper states.")
        states["home"][GRIPPER_JOINT] = gripper_states["closed"]
        states["rest"][GRIPPER_JOINT] = gripper_states["open"]
        return states

    @staticmethod
    def _start_process(command: list[str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> int:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                return process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
        return process.wait(timeout=5)

    def _refresh_processes_locked(self) -> None:
        if self._record_process is not None:
            code = self._record_process.poll()
            if code is not None:
                self._record_process = None
                if code != 0:
                    self._last_error = f"rosbag2 recorder exited with code {code}."
        if self._replay_process is not None:
            code = self._replay_process.poll()
            if code is not None:
                self._replay_process = None
                self._replay_outcome = "completed" if code == 0 else "failed"
                if code != 0:
                    self._last_error = f"rosbag2 replay exited with code {code}."
                self._enter_idle_locked()

    def close(self) -> None:
        self._http_server.shutdown()
        self._http_server.server_close()
        self._http_thread.join(timeout=2)
        with self._lock:
            processes = [self._record_process, self._replay_process]
            self._record_process = None
            self._replay_process = None
        for process in processes:
            if process is not None:
                self._stop_process(process)


def main() -> None:
    rclpy.init()
    node = TriskelOperator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
