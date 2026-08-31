"""Launch the complete Triskel stack and its ROS-native operator dashboard."""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def _servo_executable() -> str:
    """Resolve the installed standalone Servo executable across MoveIt releases."""

    executable_directory = Path(get_package_prefix("moveit_servo")) / "lib" / "moveit_servo"
    for candidate in ("servo_node_main", "servo_node"):
        if (executable_directory / candidate).is_file():
            return candidate
    raise RuntimeError(f"No standalone MoveIt Servo executable found in {executable_directory}")


def generate_launch_description() -> LaunchDescription:
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")
    start_rviz = LaunchConfiguration("start_rviz")
    dashboard_host = LaunchConfiguration("dashboard_host")
    dashboard_port = LaunchConfiguration("dashboard_port")
    recording_root = LaunchConfiguration("recording_root")
    start_visualizer = LaunchConfiguration("start_visualizer")
    visualizer_host = LaunchConfiguration("visualizer_host")
    visualizer_port = LaunchConfiguration("visualizer_port")

    bringup_share = Path(get_package_share_directory("triskel_bringup"))
    moveit_share = Path(get_package_share_directory("triskel_moveit_config"))
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(bringup_share / "launch" / "robot.launch.py")),
        launch_arguments={
            "use_mock_hardware": use_mock_hardware,
            "serial_port": serial_port,
            "baud_rate": baud_rate,
            "start_rviz": start_rviz,
        }.items(),
    )
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(moveit_share / "launch" / "move_group.launch.py"))
    )

    moveit_config = (
        MoveItConfigsBuilder("triskel", package_name="triskel_moveit_config")
        .robot_description(file_path="config/triskel.urdf.xacro")
        .robot_description_semantic(file_path="config/triskel.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    servo_parameters = yaml.safe_load(
        (moveit_share / "config" / "servo.yaml").read_text(encoding="utf-8")
    )
    servo_node = Node(
        package="moveit_servo",
        executable=_servo_executable(),
        name="servo_node",
        parameters=[
            {"moveit_servo": servo_parameters},
            {"update_period": 0.02},
            {"planning_group_name": "arm"},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        output="screen",
    )
    runtime = ParameterValue(
        PythonExpression(["'sim' if '", use_mock_hardware, "'.lower() == 'true' else 'real'"]),
        value_type=str,
    )
    operator_node = Node(
        package="triskel_operator",
        executable="dashboard",
        name="triskel_operator",
        parameters=[
            {
                "runtime": runtime,
                "host": dashboard_host,
                "port": ParameterValue(dashboard_port, value_type=int),
                "recording_root": recording_root,
                "visualization_enabled": ParameterValue(start_visualizer, value_type=bool),
                "visualization_port": ParameterValue(visualizer_port, value_type=int),
            }
        ],
        output="screen",
    )
    visualizer_node = Node(
        package="triskel_visualization",
        executable="visualizer",
        name="triskel_visualizer",
        parameters=[
            {
                "host": visualizer_host,
                "port": ParameterValue(visualizer_port, value_type=int),
            }
        ],
        condition=IfCondition(start_visualizer),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_mock_hardware", default_value="true"),
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("baud_rate", default_value="1000000"),
            DeclareLaunchArgument("start_rviz", default_value="false"),
            DeclareLaunchArgument("dashboard_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("dashboard_port", default_value="8765"),
            DeclareLaunchArgument("start_visualizer", default_value="true"),
            DeclareLaunchArgument("visualizer_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("visualizer_port", default_value="8080"),
            DeclareLaunchArgument("recording_root", default_value="~/.ros/triskel/recordings"),
            robot_launch,
            move_group_launch,
            servo_node,
            visualizer_node,
            operator_node,
        ]
    )
