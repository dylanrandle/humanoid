"""MeshCat visualizer wrapper for robot visualization.

This module provides a convenient wrapper around Pinocchio's MeshcatVisualizer
that integrates seamlessly with the Robot base class.
"""

import copy
from typing import Any

import meshcat.geometry as g
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer as PinocchioMeshcatVisualizer

from humanoid.logger import get_logger
from humanoid.robots.base import Robot
from humanoid.types.visualizer import VisualizerConfig

logger = get_logger(__name__)


class JointCommandVisualizer:
    """Visualizer for commanded joint positions (ghost robot).

    This creates a semi-transparent copy of the robot to show commanded
    joint positions alongside the actual robot state.
    """

    def __init__(self, robot: Robot, viewer: PinocchioMeshcatVisualizer, opacity: float = 0.3):
        """Initialize the joint command visualizer.

        Args:
            robot: Robot instance to visualize
            viewer: Parent MeshcatVisualizer instance
            opacity: Opacity of the ghost robot (0.0 = transparent, 1.0 = opaque)
        """
        self.robot = robot
        self._parent_viewer = viewer
        self._opacity = opacity
        self._viewer: PinocchioMeshcatVisualizer | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the joint command visualizer with a semi-transparent robot model."""

        visual_model_copy = copy.deepcopy(self.robot.visual_model)

        # Set opacity and color for all visual geometries
        for geom in visual_model_copy.geometryObjects:
            # meshColor is an RGBA array [R, G, B, A]
            # Set to green color with specified opacity
            geom.meshColor = np.array([0.0, 1.0, 0.0, self._opacity])  # Green with opacity
            # Force Pinocchio to use this color/opacity instead of the URDF's default
            geom.overrideMaterial = True

        # Create a separate visualizer for the commanded positions
        self._viewer = PinocchioMeshcatVisualizer(
            self.robot.model,
            self.robot.collision_model,
            visual_model_copy,
        )

        # Initialize with the parent viewer's meshcat instance
        # Use a different prefix to avoid conflicts
        self._viewer.initViewer(viewer=self._parent_viewer.viewer)
        self._viewer.loadViewerModel(rootNodeName="robot_joint_command")

        # Hide collisions for the joint command robot
        self._viewer.displayCollisions(False)

        self._initialized = True

    def display(self, q: np.ndarray) -> None:
        """Update the joint command visualizer with new joint positions.

        Args:
            q: Joint configuration vector (must match robot.model.nq)
        """
        if not self._initialized:
            raise RuntimeError("Joint command visualizer not initialized. Call initialize() first.")

        if len(q) != self.robot.model.nq:
            raise ValueError(
                f"Configuration vector has incorrect size. "
                f"Expected {self.robot.model.nq}, got {len(q)}"
            )

        self._viewer.display(q)  # type: ignore[union-attr]

    def set_visible(self, visible: bool) -> None:
        """Toggle visibility of the joint command robot.

        Args:
            visible: Whether the joint command robot should be visible
        """
        if not self._initialized:
            raise RuntimeError("Joint command visualizer not initialized. Call initialize() first.")

        self._viewer.displayVisuals(visible)  # type: ignore[union-attr]


class ToolCommandVisualizer:
    """Visualizer for commanded tool poses (end effector ghost).

    This creates a semi-transparent visualization of the end effector
    (and all its child links) at the commanded tool pose.
    """

    def __init__(
        self,
        robot: Robot,
        viewer: PinocchioMeshcatVisualizer,
        end_effector_frame: str,
        opacity: float = 0.5,
    ):
        """Initialize the tool command visualizer.

        Args:
            robot: Robot instance to visualize
            viewer: Parent MeshcatVisualizer instance
            end_effector_frame: Name of the end effector frame
            opacity: Opacity of the ghost end effector (0.0 = transparent, 1.0 = opaque)
        """
        self.robot = robot
        self._parent_viewer = viewer
        self._end_effector_frame = end_effector_frame
        self._opacity = opacity
        self._viewer: PinocchioMeshcatVisualizer | None = None
        self._initialized = False
        self._end_effector_frame_id: int | None = None

    def _get_end_effector_subtree_joints(self) -> set[int]:
        """Get all joint indices that are part of the end effector subtree.

        Returns:
            Set of joint indices that belong to the end effector and its children
        """
        # Get the end effector frame
        ee_frame_id = self.robot.get_frame_id(self._end_effector_frame)
        ee_frame = self.robot.model.frames[ee_frame_id]

        # Start with the parent joint of the end effector frame
        subtree_joints = {ee_frame.parentJoint}

        # Find all descendant joints using the kinematic tree
        # We'll traverse the tree starting from the end effector's parent joint
        def add_descendants(joint_id: int):
            """Recursively add all descendant joints."""
            for i in range(1, self.robot.model.njoints):
                parent = self.robot.model.parents[i]
                if parent == joint_id and i not in subtree_joints:
                    subtree_joints.add(i)
                    add_descendants(i)

        add_descendants(ee_frame.parentJoint)

        logger.debug(f"End effector subtree joints: {subtree_joints}")
        return subtree_joints

    def initialize(self) -> None:
        """Initialize the tool command visualizer with end effector geometries only."""
        # Get the end effector frame ID
        self._end_effector_frame_id = self.robot.get_frame_id(self._end_effector_frame)

        # Get all joints that are part of the end effector subtree
        ee_joints = self._get_end_effector_subtree_joints()

        # Create a copy of the visual model with only end effector geometries
        visual_model_copy = pin.GeometryModel()

        # Filter geometries that belong to the end effector subtree
        for geom in self.robot.visual_model.geometryObjects:
            # Check if this geometry's parent joint is in the end effector subtree
            if geom.parentJoint in ee_joints:
                # Create a copy of the geometry
                geom_copy = geom.clone()
                # Set color to blue with specified opacity
                geom_copy.meshColor = np.array([0.0, 0.5, 1.0, self._opacity])  # Blue with opacity
                geom_copy.overrideMaterial = True
                visual_model_copy.addGeometryObject(geom_copy)

        logger.info(
            f"Tool command visualizer: {len(visual_model_copy.geometryObjects)} "
            f"geometries from end effector subtree"
        )

        # Create an empty collision model (we don't need collisions for visualization)
        collision_model_copy = pin.GeometryModel()

        # Create a separate visualizer for the tool command
        self._viewer = PinocchioMeshcatVisualizer(
            self.robot.model,
            collision_model_copy,
            visual_model_copy,
        )

        # Initialize with the parent viewer's meshcat instance
        self._viewer.initViewer(viewer=self._parent_viewer.viewer)
        self._viewer.loadViewerModel(rootNodeName="tool_command")

        # Hide collisions for the tool command robot
        self._viewer.displayCollisions(False)

        self._initialized = True

    def display(self, tool_pose: pin.SE3) -> None:
        """Update the tool command visualizer with a new commanded tool pose.

        This computes the inverse kinematics to find a configuration that places
        the end effector at the commanded pose, then displays the end effector
        geometry at that configuration.

        Args:
            tool_pose: Commanded SE3 pose for the end effector
        """
        if not self._initialized:
            raise RuntimeError("Tool command visualizer not initialized. Call initialize() first.")

        # We need to compute a configuration that places the end effector at the commanded pose
        # For visualization purposes, we can use the current robot configuration as a starting point
        # and just update the end effector frame's placement

        # Get the current configuration (we'll use this as a base)
        q = (
            self.robot.data.q.copy()
            if hasattr(self.robot.data, "q")
            else np.zeros(self.robot.model.nq)
        )

        # Update forward kinematics
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        # Display the robot at this configuration
        # The actual positioning will be handled by updating the root transform
        self._viewer.display(q)  # type: ignore[union-attr]

        # Now we need to adjust the root transform so the end effector appears at tool_pose
        # Get the current end effector pose in the robot's configuration
        ee_pose_in_config = self.robot.data.oMf[self._end_effector_frame_id]

        # Compute the transform needed to move the end effector to the commanded pose
        # tool_pose = root_transform * ee_pose_in_config
        # => root_transform = tool_pose * ee_pose_in_config^-1
        root_transform = tool_pose * ee_pose_in_config.inverse()

        # Apply this transform to the entire visualization
        self._viewer.viewer["tool_command"].set_transform(root_transform.homogeneous)  # type: ignore[union-attr]

    def set_visible(self, visible: bool) -> None:
        """Toggle visibility of the tool command visualization.

        Args:
            visible: Whether the tool command visualization should be visible
        """
        if not self._initialized:
            raise RuntimeError("Tool command visualizer not initialized. Call initialize() first.")

        self._viewer.displayVisuals(visible)  # type: ignore[union-attr]


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
        self._joint_command_viz: JointCommandVisualizer | None = None
        self._tool_command_viz: ToolCommandVisualizer | None = None
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

        # Initialize joint command visualizer if enabled
        if self._config.show_commanded_joint_positions:
            self._joint_command_viz = JointCommandVisualizer(
                self.robot, self._viewer, opacity=self._config.joint_command_opacity
            )
            self._joint_command_viz.initialize()

        # Initialize tool command visualizer if enabled
        if self._config.show_commanded_tool_pose:
            self._tool_command_viz = ToolCommandVisualizer(
                self.robot,
                self._viewer,
                end_effector_frame=self.robot.config.end_effector_frame,
                opacity=self._config.tool_command_opacity,
            )
            self._tool_command_viz.initialize()

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

    def display_joint_command(self, q: np.ndarray) -> None:
        """Update the joint command visualizer with commanded joint positions.

        This displays a semi-transparent "ghost" robot showing the commanded
        joint positions, useful for visualizing the difference between commanded
        and actual positions.

        Args:
            q: Commanded joint configuration vector (must match robot.model.nq)

        Raises:
            RuntimeError: If the visualizer or joint command visualizer has not been initialized
            ValueError: If q has incorrect dimensions
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        if self._joint_command_viz is None:
            raise RuntimeError(
                "Joint command visualizer not enabled. "
                "Set show_commanded_joint_positions=True in config."
            )

        if len(q) != self.robot.model.nq:
            raise ValueError(
                f"Configuration vector has incorrect size. "
                f"Expected {self.robot.model.nq}, got {len(q)}"
            )

        self._joint_command_viz.display(q)

    def display_tool_command(self, tool_pose: pin.SE3) -> None:
        """Update the tool command visualizer with a commanded tool pose.

        This displays a semi-transparent "ghost" end effector showing the commanded
        tool pose, useful for visualizing the target pose for the end effector.

        Args:
            tool_pose: Commanded SE3 pose for the end effector

        Raises:
            RuntimeError: If the visualizer or tool command visualizer has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")

        if self._tool_command_viz is None:
            raise RuntimeError(
                "Tool command visualizer not enabled. Set show_commanded_tool_pose=True in config."
            )

        self._tool_command_viz.display(tool_pose)

    def set_joint_command_visible(self, visible: bool) -> None:
        """Toggle visibility of the joint command visualization.

        Args:
            visible: Whether the joint command robot should be visible

        Raises:
            RuntimeError: If the joint command visualizer has not been initialized
        """
        if self._joint_command_viz is None:
            raise RuntimeError(
                "Joint command visualizer not enabled. "
                "Set show_commanded_joint_positions=True in config."
            )
        self._joint_command_viz.set_visible(visible)

    def set_tool_command_visible(self, visible: bool) -> None:
        """Toggle visibility of the tool command visualization.

        Args:
            visible: Whether the tool command visualization should be visible

        Raises:
            RuntimeError: If the tool command visualizer has not been initialized
        """
        if self._tool_command_viz is None:
            raise RuntimeError(
                "Tool command visualizer not enabled. Set show_commanded_tool_pose=True in config."
            )
        self._tool_command_viz.set_visible(visible)

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
