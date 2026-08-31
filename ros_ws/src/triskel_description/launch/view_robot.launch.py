"""Display the canonical Triskel model without starting a control stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_gui = LaunchConfiguration("use_gui")
    start_rviz = LaunchConfiguration("start_rviz")
    description_share = FindPackageShare("triskel_description")
    urdf_path = PathJoinSubstitution([description_share, "urdf", "triskel.urdf"])
    rviz_path = PathJoinSubstitution([description_share, "rviz", "description.rviz"])
    robot_description = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", urdf_path]),
        value_type=str,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_gui", default_value="true"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                condition=IfCondition(use_gui),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_path],
                condition=IfCondition(start_rviz),
                output="screen",
            ),
        ]
    )
