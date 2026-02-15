"""MeshCat visualizer wrapper for robot visualization.

This module provides a convenient wrapper around Pinocchio's MeshcatVisualizer
that integrates seamlessly with the Robot base class.
"""

from typing import Any

import meshcat.geometry as g
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer as PinocchioMeshcatVisualizer

from humanoid.robots.base import Robot
from humanoid.types.visualizer import VisualizerConfig


class MeshcatVisualizer:
    """Wrapper around Pinocchio's MeshcatVisualizer for easy robot visualization.

    This class simplifies the process of visualizing robots by automatically
    handling the initialization and configuration of the MeshCat visualizer
    based on a Robot instance.

    Example:
        >>> from humanoid.robots.base import Robot
        >>> from humanoid.visualizers.meshcat import MeshcatVisualizer
        >>> from humanoid.types.visualizer import VisualizerConfig
        >>>
        >>> # Load robot
        >>> robot = Robot.from_name("panda")
        >>>
        >>> # Create and initialize visualizer
        >>> config = VisualizerConfig(open_browser=True, show_collisions=False)
        >>> viz = MeshcatVisualizer(robot, config=config)
        >>> viz.initialize()
        >>>
        >>> # Update robot configuration
        >>> q = np.zeros(robot.model.nq)
        >>> viz.display(q)

    Attributes:
        robot: The Robot instance being visualized
        viewer: The underlying Pinocchio MeshcatVisualizer instance
    """

    def __init__(self, robot: Robot, config: VisualizerConfig):
        """Initialize the MeshCat visualizer wrapper.

        Args:
            robot: Robot instance to visualize
            config: Visualizer configuration (controls browser opening, collision display, etc.)
        """
        self.robot = robot
        self._config = config
        self._viewer: PinocchioMeshcatVisualizer | None = None
        self._initialized = False

    @property
    def viewer(self) -> PinocchioMeshcatVisualizer:
        """Get the underlying Pinocchio MeshcatVisualizer.

        Returns:
            The Pinocchio MeshcatVisualizer instance

        Raises:
            RuntimeError: If the visualizer has not been initialized
        """
        if self._viewer is None:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")
        return self._viewer

    def initialize(self) -> None:
        """Initialize the MeshCat visualizer and load the robot model.

        This method creates the visualizer, opens the viewer,
        and loads the robot's visual and collision models.
        """
        # Create the Pinocchio MeshcatVisualizer
        self._viewer = PinocchioMeshcatVisualizer(
            self.robot.model,
            self.robot.collision_model,
            self.robot.visual_model,
        )

        # Initialize the viewer
        self._viewer.initViewer(open=self._config.open_browser)

        # Load the robot model into the viewer
        self._viewer.loadViewerModel()

        # Configure collision visibility
        self._viewer.displayCollisions(self._config.show_collisions)

        self._initialized = True

    def display(self, q: np.ndarray) -> None:
        """Update the visualizer with a new robot configuration.

        Args:
            q: Joint configuration vector (must match robot.model.nq)

        Raises:
            RuntimeError: If the visualizer has not been initialized
            ValueError: If q has incorrect dimensions
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        if len(q) != self.robot.model.nq:
            raise ValueError(
                f"Configuration vector has incorrect size. "
                f"Expected {self.robot.model.nq}, got {len(q)}"
            )

        self.viewer.display(q)

    def display_collisions(self, visible: bool) -> None:
        """Toggle visibility of collision geometries.

        Args:
            visible: Whether collision geometries should be visible

        Raises:
            RuntimeError: If the visualizer has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        self.viewer.displayCollisions(visible)

    def display_visuals(self, visible: bool) -> None:
        """Toggle visibility of visual geometries.

        Args:
            visible: Whether visual geometries should be visible

        Raises:
            RuntimeError: If the visualizer has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        self.viewer.displayVisuals(visible)

    def get_url(self) -> str:
        """Get the URL of the MeshCat visualizer.

        Returns:
            URL string where the visualizer can be accessed

        Raises:
            RuntimeError: If the visualizer has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        return self.viewer.viewer.url()

    def add_sphere(
        self,
        name: str,
        position: np.ndarray,
        radius: float = 0.01,
        color: int = 0xFF0000,
        opacity: float = 1.0,
    ) -> None:
        """Add a sphere marker to the visualization.

        Useful for visualizing target positions, waypoints, or other points of interest.

        Args:
            name: Unique name for the sphere (use "/" for hierarchy, e.g., "targets/sphere_1")
            position: 3D position vector [x, y, z]
            radius: Radius of the sphere in meters (default: 0.01m = 1cm)
            color: RGB color as hex integer (default: 0xFF0000 = red)
            opacity: Opacity from 0.0 (transparent) to 1.0 (opaque)

        Raises:
            RuntimeError: If the visualizer has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        self.viewer.viewer[name].set_object(
            g.Sphere(radius),
            g.MeshBasicMaterial(color=color, opacity=opacity),
        )
        transform = pin.SE3(np.eye(3), position).homogeneous
        self.viewer.viewer[name].set_transform(transform)

    def add_frame(
        self,
        name: str,
        pose: pin.SE3,
        axis_length: float = 0.1,
    ) -> None:
        """Add a coordinate frame marker to the visualization.

        Displays RGB axes (X=red, Y=green, Z=blue) at the specified pose.

        Args:
            name: Unique name for the frame (use "/" for hierarchy)
            pose: SE3 transformation representing the frame pose
            axis_length: Length of each axis in meters (default: 0.1m = 10cm)

        Raises:
            RuntimeError: If the visualizer has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        self.viewer.viewer[name].set_object(
            g.triad(scale=axis_length),
        )
        self.viewer.viewer[name].set_transform(pose.homogeneous)

    def remove_object(self, name: str) -> None:
        """Remove an object from the visualization.

        Args:
            name: Name of the object to remove

        Raises:
            RuntimeError: If the visualizer has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        self.viewer.viewer[name].delete()

    def __getitem__(self, name: str) -> Any:
        """Access the underlying MeshCat viewer's scene tree.

        This allows direct access to the MeshCat API for advanced usage.

        Args:
            name: Path in the scene tree

        Returns:
            MeshCat scene tree node

        Raises:
            RuntimeError: If the visualizer has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        return self.viewer.viewer[name]
