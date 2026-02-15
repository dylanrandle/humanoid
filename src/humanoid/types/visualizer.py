from dataclasses import dataclass


@dataclass
class VisualizerConfig:
    open_browser: bool = True
    show_collisions: bool = False
