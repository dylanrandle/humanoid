"""Start MoveIt move_group against an already-running Triskel bringup."""

from launch import LaunchDescription
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def generate_launch_description() -> LaunchDescription:
    moveit_config = (
        MoveItConfigsBuilder(
            "triskel",
            package_name="triskel_moveit_config",
        )
        .planning_pipelines(pipelines=["ompl"])
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            # robot_state_publisher is the single authority for /robot_description.
            # Publishing MoveIt's planning-only URDF here would overwrite the
            # hardware-enabled description consumed by ros2_control.
            publish_robot_description=False,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )
    return generate_move_group_launch(moveit_config)
