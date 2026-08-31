"""Bring up Triskel through robot_state_publisher and ros2_control."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_rviz = LaunchConfiguration("start_rviz")

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

    def controller_spawner(name: str, *extra_arguments: str) -> Node:
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "--controller-manager", "/controller_manager", *extra_arguments],
            output="screen",
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_mock_hardware",
                default_value="true",
                description="Use ros2_control GenericSystem instead of the physical Feetech bus.",
            ),
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("baud_rate", default_value="1000000"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("start_rviz", default_value="false"),
            robot_state_publisher,
            controller_manager,
            controller_spawner("joint_state_broadcaster"),
            controller_spawner(
                "omni_base_controller",
                "--controller-ros-args",
                "--ros-args --remap ~/cmd_vel:=/cmd_vel",
            ),
            controller_spawner("arm_controller"),
            controller_spawner("gripper_controller"),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_path],
                condition=IfCondition(start_rviz),
                output="screen",
            ),
        ]
    )
