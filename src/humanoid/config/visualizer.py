"""Visualizer configuration defaults."""

from humanoid.types.visualizer import VisualizerConfig

VISUALIZER_CONFIG = VisualizerConfig()

# Viser's light-theme background and add_grid defaults.
VIEWER_BACKGROUND: tuple[float, float, float] = (1, 1, 1)
EMPTY_GRID_CELL_COLOR: tuple[int, int, int] = (200, 200, 200)
EMPTY_GRID_SECTION_COLOR: tuple[int, int, int] = (140, 140, 140)
EMPTY_GRID_CELL_SIZE = 0.5
EMPTY_GRID_SECTION_SIZE = 1.0
# Classic MuJoCo approximates Viser's HDR environment with ambient fill.
VIEWER_LIGHT_AMBIENT: tuple[float, float, float] = (0.4, 0.4, 0.4)
# Convert Viser's Y-up light positions to Z-up, then point toward the origin.
VIEWER_KEY_DIRECTION: tuple[float, float, float] = (0.2, -0.2, -1)
# The classic renderer uses this origin for its directional-light shadow camera.
VIEWER_KEY_POSITION: tuple[float, float, float] = (-0.4, 1, 2)
VIEWER_FILL_DIRECTION: tuple[float, float, float] = (0, 0, 1)
# Preserve Viser's 3.0 : 0.4 key/fill ratio at a classic-renderer exposure.
VIEWER_KEY_INTENSITY = 0.65
VIEWER_FILL_INTENSITY = VIEWER_KEY_INTENSITY * 0.4 / 3
