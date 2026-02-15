"""Base robot class for loading robot models in Pinocchio."""

from pathlib import Path

import numpy as np
import pinocchio as pin

from humanoid.logger import get_logger

COLLISION_URDF_SUFFIX = "_collision.urdf"


logger = get_logger(__name__)


class Robot:
    """Base class for loading and managing robot models in Pinocchio.

    This class provides a standardized interface for loading URDF robot models
    with collision and visual geometries using Pinocchio.

    Attributes:
        name: Name of the robot (e.g., "panda")
        model: Pinocchio model containing kinematic and dynamic information
        collision_model: Pinocchio collision model
        visual_model: Pinocchio visual model
        data: Pinocchio data structure for computations
        collision_data: Pinocchio collision data
        visual_data: Pinocchio visual data
    """

    def __init__(
        self,
        name: str,
        urdf_filename: str | None = None,
        collision_urdf_filename: str | None = None,
        package_dirs: list[Path] | None = None,
    ):
        """Initialize the robot by loading its URDF model.

        Args:
            name: Name of the robot (should match the directory name in assets/)
            urdf_filename: Name of the URDF file for visual model. If None, uses "{name}.urdf"
            collision_urdf_filename: Optional separate URDF for collision model.
                                    If None, automatically checks one
                                    and falls back to urdf_filename if not found.
            package_dirs: Additional package directories for mesh loading.
                         The assets directory is automatically included.
        """
        self.name = name

        # Determine paths relative to this file
        robots_dir = Path(__file__).parent
        assets_dir = robots_dir / "assets"
        robot_dir = assets_dir / name

        # Default URDF filename
        if urdf_filename is None:
            urdf_filename = f"{name}.urdf"

        urdf_path = robot_dir / "urdf" / urdf_filename

        # Store paths for external access
        self._urdf_path = urdf_path
        self._robot_dir = robot_dir
        self._package_dirs = package_dirs

        # Validate paths
        if not robot_dir.exists():
            raise FileNotFoundError(
                f"Robot directory not found: {robot_dir}\n"
                f"Available robots: {[d.name for d in assets_dir.iterdir() if d.is_dir()]}"
            )

        if not urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file not found: {urdf_path}\n"
                f"Available URDF files: {list((robot_dir / 'urdf').glob('*.urdf'))}"
            )

        # Auto-detect collision URDF if not specified
        if collision_urdf_filename is None:
            # Check if a collision URDF variant exists
            collision_candidate = robot_dir / "urdf" / f"{name}{COLLISION_URDF_SUFFIX}"
            if collision_candidate.exists():
                collision_urdf_filename = f"{name}{COLLISION_URDF_SUFFIX}"
                logger.info(f"Using separate collision URDF: {collision_urdf_filename}")

        # Setup package directories for mesh loading
        if package_dirs is None:
            package_dirs = []

        # Add the robot's asset directory and the assets root directory
        package_dirs = [robot_dir, assets_dir, *package_dirs]
        package_dirs_str = [str(p) for p in package_dirs]

        # Load the kinematic model and visual model from main URDF
        self.model, self.visual_model = pin.buildModelsFromUrdf(
            str(urdf_path),
            package_dirs=package_dirs_str,
            geometry_types=[pin.GeometryType.VISUAL],
        )

        # Load collision model from separate URDF if available, otherwise use main URDF
        if collision_urdf_filename is not None:
            collision_urdf_path = robot_dir / "urdf" / collision_urdf_filename
            if not collision_urdf_path.exists():
                raise FileNotFoundError(f"Collision URDF file not found: {collision_urdf_path}")
            _, self.collision_model = pin.buildModelsFromUrdf(
                str(collision_urdf_path),
                package_dirs=package_dirs_str,
                geometry_types=[pin.GeometryType.COLLISION],
            )
        else:
            # Use the same URDF for collision
            _, self.collision_model = pin.buildModelsFromUrdf(
                str(urdf_path),
                package_dirs=package_dirs_str,
                geometry_types=[pin.GeometryType.COLLISION],
            )

        # Store final package directories
        self._package_dirs = package_dirs

        # Create data structures for computations
        self.data = self.model.createData()
        self.collision_data = pin.GeometryData(self.collision_model)
        self.visual_data = pin.GeometryData(self.visual_model)

    @property
    def urdf_path(self) -> Path:
        """Get the path to the robot's URDF file.

        Returns:
            Path to the URDF file
        """
        return self._urdf_path

    @property
    def robot_dir(self) -> Path:
        """Get the robot's asset directory.

        Returns:
            Path to the robot's asset directory
        """
        return self._robot_dir

    @property
    def package_dirs(self) -> list[Path] | None:
        """Get the package directories used for mesh loading.

        Returns:
            List of package directory paths
        """
        return self._package_dirs

    @property
    def joint_names(self) -> list[str]:
        """Get the list of joint names in the robot model.

        Returns:
            List of joint names (excluding the universe joint)
        """
        return [self.model.names[i] for i in range(1, self.model.njoints)]

    def print_info(self) -> None:
        """Print information about the robot model."""
        print(f"Robot: {self.name}")
        print(f"Model name: {self.model.name}")
        print(f"Number of joints (nq): {self.model.nq}")
        print(f"Number of velocity coordinates (nv): {self.model.nv}")
        print(f"Number of configuration variables: {self.model.nq}")
        print(f"Available frames ({len(self.model.frames)}):")
        for i, frame in enumerate(self.model.frames):
            print(f"  {i}: {frame.name}")

    def get_frame_id(self, frame_name: str) -> int:
        """Get the frame ID for a given frame name.

        Args:
            frame_name: Name of the frame

        Returns:
            Frame ID

        Raises:
            ValueError: If frame name is not found
        """
        if not self.model.existFrame(frame_name):
            available_frames = [frame.name for frame in self.model.frames]
            raise ValueError(
                f"Frame '{frame_name}' not found in model.\nAvailable frames: {available_frames}"
            )
        return self.model.getFrameId(frame_name)

    def forward_kinematics(self, q: np.ndarray) -> None:
        """Compute forward kinematics for the given configuration.

        Args:
            q: Joint configuration vector
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def get_frame_pose(self, frame_name: str, q: np.ndarray) -> pin.SE3:
        """Get the pose of a frame in the world frame.

        Args:
            frame_name: Name of the frame
            q: Joint configuration vector

        Returns:
            SE3 transformation representing the frame pose
        """
        self.forward_kinematics(q)
        frame_id = self.get_frame_id(frame_name)
        return self.data.oMf[frame_id]

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "Robot":
        """Create a robot instance from a robot name.

        This is a convenience factory method that automatically detects
        and uses collision URDFs if available.

        Args:
            name: Name of the robot (e.g., "panda")
            **kwargs: Additional arguments passed to __init__

        Returns:
            Robot instance
        """
        return cls(name=name, **kwargs)
