"""Shared controller-spawner configuration for Triskel launch files."""

from typing import Any


def controller_spawner_options() -> dict[str, Any]:
    """Return one grouped spawner configuration for every Triskel controller."""

    return {
        "package": "controller_manager",
        "executable": "spawner",
        "name": "triskel_controller_spawner",
        "arguments": [
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--service-call-timeout",
            "60",
            "--switch-timeout",
            "60",
            "--activate-as-group",
            "--controller",
            "joint_state_broadcaster",
            "--controller",
            "omni_base_controller",
            "--controller-ros-args",
            "--ros-args --remap ~/cmd_vel:=/cmd_vel",
            "--controller",
            "arm_controller",
            "--controller",
            "gripper_controller",
        ],
        "output": "screen",
    }
