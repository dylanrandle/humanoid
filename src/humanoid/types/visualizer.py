from dataclasses import dataclass


@dataclass
class VisualizerConfig:
    dt: float = 0.02
    open_browser: bool = False
    show_collisions: bool = False
    show_commanded_joint_positions: bool = True
    joint_command_opacity: float = 0.3
    show_commanded_tool_pose: bool = True
    tool_command_opacity: float = 0.5
    show_commanded_base_pose: bool = True
