"""Base robot class for loading robot models in Pinocchio."""

from pathlib import Path

import numpy as np
import pinocchio as pin

from humanoid.logger import get_logger
from humanoid.types.robot import RobotConfig

COLLISION_URDF_SUFFIX = "_collision.urdf"

# Pinocchio represents continuous (unbounded) revolute joints — JointModelRUBX/Y/Z
# and JointModelRevoluteUnboundedUnaligned — with nq=2 (storing [cos θ, sin θ])
# and nv=1. This combination uniquely identifies the family across all axes.
REVOLUTE_UNBOUNDED_NQ = 2
REVOLUTE_UNBOUNDED_NV = 1


logger = get_logger(__name__)


class Robot:
    """Base class for loading and managing robot models in Pinocchio.

    This class provides a standardized interface for loading URDF robot models
    with collision and visual geometries using Pinocchio.

    Attributes:
        config: Robot configuration containing name and other parameters
        model: Pinocchio model containing kinematic and dynamic information
        collision_model: Pinocchio collision model
        visual_model: Pinocchio visual model
        data: Pinocchio data structure for computations
        collision_data: Pinocchio collision data
        visual_data: Pinocchio visual data
    """

    def __init__(
        self,
        config: RobotConfig,
        urdf_filename: str | None = None,
        collision_urdf_filename: str | None = None,
        srdf_filename: str | None = None,
        package_dirs: list[Path] | None = None,
    ):
        """Initialize the robot by loading its URDF model.

        Args:
            config: Robot configuration containing name and other parameters
            urdf_filename: Name of the URDF file for visual model. Default: "{config.name}.urdf"
            collision_urdf_filename: Optional separate URDF for collision model.
                                    If None, automatically checks one
                                    and falls back to urdf_filename if not found.
            package_dirs: Additional package directories for mesh loading.
                         The assets directory is automatically included.
        """
        self._config = config

        # Determine paths relative to this file
        robots_dir = Path(__file__).parent
        assets_dir = robots_dir / "assets"
        robot_dir = assets_dir / config.name

        # Default URDF filename
        if urdf_filename is None:
            urdf_filename = f"{config.name}.urdf"

        if srdf_filename is None:
            srdf_filename = f"{config.name}.srdf"

        urdf_path = robot_dir / "urdf" / urdf_filename
        srdf_path = robot_dir / "srdf" / srdf_filename

        # Store paths for external access
        self._urdf_path = urdf_path
        self._srdf_path = srdf_path
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
            collision_candidate = robot_dir / "urdf" / f"{config.name}{COLLISION_URDF_SUFFIX}"
            if collision_candidate.exists():
                collision_urdf_filename = f"{config.name}{COLLISION_URDF_SUFFIX}"
                logger.info(f"Using separate collision URDF: {collision_urdf_filename}")

        # Setup package directories for mesh loading
        if package_dirs is None:
            package_dirs = []

        # Add the robot's asset directory and the assets root directory
        package_dirs = [robot_dir, assets_dir, *package_dirs]
        package_dirs_str = [str(p) for p in package_dirs]

        # Add a planar joint if a base_frame is specified
        root_joint = pin.JointModelPlanar() if config.base_frame else None

        # Load the kinematic model and visual model from main URDF
        self.model, self.visual_model = pin.buildModelsFromUrdf(
            str(urdf_path),
            package_dirs=package_dirs_str,
            geometry_types=[pin.GeometryType.VISUAL],
            root_joint=root_joint,
            mimic=True,
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
                root_joint=root_joint,
                mimic=True,
            )
        else:
            # Use the same URDF for collision
            _, self.collision_model = pin.buildModelsFromUrdf(
                str(urdf_path),
                package_dirs=package_dirs_str,
                geometry_types=[pin.GeometryType.COLLISION],
                root_joint=root_joint,
                mimic=True,
            )

        # Store final package directories
        self._package_dirs = package_dirs

        # Create data structures for computations
        self.data = self.model.createData()
        self.collision_data = pin.GeometryData(self.collision_model)
        self.visual_data = pin.GeometryData(self.visual_model)

    @property
    def config(self) -> RobotConfig:
        """Get the robot configuration.

        Returns:
            Robot configuration
        """
        return self._config

    @property
    def urdf_path(self) -> Path:
        """Get the path to the robot's URDF file.

        Returns:
            Path to the URDF file
        """
        return self._urdf_path

    @property
    def srdf_path(self) -> Path:
        """Get the path to the robot's SRDF file.

        Returns:
            Path to the SRDF file
        """
        return self._srdf_path

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
        print(f"Model name: {self.model.name}")
        # NOTE: excludes universe joint
        print(f"Number of joints: {len(self.model.joints) - 1}")
        print(f"Number of configuration variables (nq): {self.model.nq}")
        print(f"Number of velocity coordinates (nv): {self.model.nv}")

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

    def assert_frame_exists(self, frame_name: str) -> bool:
        if not self.model.existFrame(frame_name):
            raise ValueError(f"Frame '{frame_name}' not found in URDF")
        return True

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

    def joint_positions_to_q(self, joint_idx_to_position: dict[int, float]) -> np.ndarray:
        """Convert servo positions to a Pinocchio configuration vector q.

        Joints with no entry (e.g., a planar base joint added as root_joint) are left
        at neutral. Continuous (revolute-unbounded) joints have nq=2; their angle θ is
        stored as [cos(θ), sin(θ)].

        Args:
            joint_idx_to_position: 0-based joint index → servo position (rad)

        Returns:
            q: Configuration vector of length model.nq
        """
        q = pin.neutral(self.model)
        for joint_idx, position in joint_idx_to_position.items():
            joint = self.model.joints[joint_idx + 1]
            if joint.nq == REVOLUTE_UNBOUNDED_NQ and joint.nv == REVOLUTE_UNBOUNDED_NV:
                q[joint.idx_q] = np.cos(position)
                q[joint.idx_q + 1] = np.sin(position)
            elif joint.nq == 1:
                q[joint.idx_q] = position
        return q

    def joint_velocities_to_v(self, joint_idx_to_velocity: dict[int, float]) -> np.ndarray:
        """Convert servo velocities to a Pinocchio velocity vector v.

        Joints with no entry (e.g., a planar base joint) are left at zero.

        Args:
            joint_idx_to_velocity: 0-based joint index → servo velocity (rad/s)

        Returns:
            v: Velocity vector of length model.nv
        """
        v = np.zeros(self.model.nv)
        for joint_idx, velocity in joint_idx_to_velocity.items():
            joint = self.model.joints[joint_idx + 1]
            if joint.nv == 1:
                v[joint.idx_v] = velocity
        return v

    def joint_idx_to_position_idx(self, joint_idx: int) -> int:
        """Return the index into the configuration vector q for the given joint.

        Args:
            joint_idx: 0-based joint index (universe joint at Pinocchio index 0 is skipped)

        Returns:
            Starting index of this joint's position coordinates in q
        """
        return self.model.joints[joint_idx + 1].idx_q

    def joint_idx_to_velocity_idx(self, joint_idx: int) -> int:
        """Return the index into the velocity vector v for the given joint.

        Args:
            joint_idx: 0-based joint index (universe joint at Pinocchio index 0 is skipped)

        Returns:
            Starting index of this joint's velocity coordinates in v
        """
        return self.model.joints[joint_idx + 1].idx_v

    def get_tool_pose(self, q: np.ndarray) -> pin.SE3:
        """Get the pose of the configured tool frame.

        Args:
            q: Joint configuration vector

        Returns:
            SE3 transformation for ``config.tool_frame`` in the world frame
        """
        return self.get_frame_pose(self.config.tool_frame, q)

    def get_base_pose(self, q: np.ndarray) -> pin.SE3 | None:
        """Get the pose of the configured base frame, or None if not configured.

        Args:
            q: Joint configuration vector

        Returns:
            SE3 transformation for ``config.base_frame`` in the world frame,
            or None if no base frame is configured.
        """
        if self.config.base_frame is None:
            return None
        return self.get_frame_pose(self.config.base_frame, q)

    def get_gripper_position_indices(self) -> list[int]:
        """Return the position indices for each configured gripper joint.

        Returns an empty list when no gripper joints are configured. Each
        joint index in ``config.gripper_joint_indices`` is mapped through
        :meth:`joint_idx_to_position_idx`, which is the correct index into
        ``q`` / observation arrays even when a planar base joint shifts the
        layout.
        """
        if not self.config.gripper_joint_indices:
            return []
        return [self.joint_idx_to_position_idx(j) for j in self.config.gripper_joint_indices]

    def get_gripper_limits(self) -> list[tuple[float, float]]:
        """Return ``(lower, upper)`` position limits for each gripper joint.

        Returns an empty list when no gripper joints are configured.
        """
        return [
            (
                float(self.model.lowerPositionLimit[idx]),
                float(self.model.upperPositionLimit[idx]),
            )
            for idx in self.get_gripper_position_indices()
        ]

    def set_gripper_positions(self, q: np.ndarray, gripper_positions: np.ndarray) -> None:
        """
        Write gripper positions into ``q`` via the joint→position index mapping.
        No-op when no gripper joints are configured.

        Args:
            q: Configuration vector mutated in place
            gripper_positions: Target positions, one per gripper joint
        """
        if not self.config.gripper_joint_indices:
            return
        assert len(gripper_positions) == len(self.config.gripper_joint_indices), (
            "Received invalid number of gripper_positions "
            f"(expected: {len(self.config.gripper_joint_indices)}, "
            f"received: {len(gripper_positions)})"
        )
        for i, joint_idx in enumerate(self.config.gripper_joint_indices):
            q[self.joint_idx_to_position_idx(joint_idx)] = gripper_positions[i]
