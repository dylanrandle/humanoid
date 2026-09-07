"""Bring up Triskel through robot_state_publisher and ros2_control."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from triskel_bringup.controller_spawning import controller_spawner_options


def generate_launch_description() -> LaunchDescription:
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_rviz = LaunchConfiguration("start_rviz")
    spawn_controllers = LaunchConfiguration("spawn_controllers")

    bringup_share = FindPackageShare("triskel_bringup")
    control_share = FindPackageShare("triskel_control")
    description_share = FindPackageShare("triskel_description")
    xacro_path = PathJoinSubstitution([bringup_share, "urdf", "triskel.urdf.xacro"])
    controllers_path = PathJoinSubstitution([control_share, "config", "controllers.yaml"])
    rviz_path = PathJoinSubstitution([description_share, "rviz", "description.rviz"])
    robot_description = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                xacro_path,
                " use_mock_hardware:=",
                use_mock_hardware,
                " serial_port:=",
                serial_port,
                " baud_rate:=",
                baud_rate,
            ]
        ),
        value_type=str,
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controllers_path, {"use_sim_time": use_sim_time}],
        remappings=[("robot_description", "/robot_description")],
        output="screen",
        on_exit=Shutdown(),
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )

    controller_spawner = Node(
        **controller_spawner_options(),
        condition=IfCondition(spawn_controllers),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_mock_hardware",
                default_value="true",
                description="Use the STS hardware interface's built-in simulation backend.",
            ),
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("baud_rate", default_value="1000000"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("start_rviz", default_value="false"),
            DeclareLaunchArgument("spawn_controllers", default_value="true"),
            robot_state_publisher,
            controller_manager,
            controller_spawner,
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_path],
                condition=IfCondition(start_rviz),
                output="screen",
            ),
        ]
    )
