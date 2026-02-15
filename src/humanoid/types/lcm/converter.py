"""Converter between LCM types and Python dataclasses."""

import numpy as np
import pinocchio as pin

from humanoid.types.lcm import robot_joint_command_t, robot_state_t, robot_tool_command_t
from humanoid.types.robot import RobotJointCommand, RobotState, RobotToolCommand


class LCMConverter:
    """Handles conversion between LCM types and Python dataclasses."""

    @staticmethod
    def robot_joint_command_to_lcm(command: RobotJointCommand) -> robot_joint_command_t:
        """Convert RobotJointCommand dataclass to robot_joint_command_t LCM type.

        Args:
            command: RobotJointCommand dataclass with timestamp and joint_positions array

        Returns:
            robot_joint_command_t LCM type ready for transmission
        """
        lcm_command = robot_joint_command_t()
        lcm_command.timestamp = int(command.timestamp * 1e9)  # Convert to nanoseconds
        lcm_command.num_joints = len(command.joint_positions)
        lcm_command.joint_positions = command.joint_positions.tolist()

        return lcm_command

    @staticmethod
    def robot_joint_command_from_lcm(lcm_command: robot_joint_command_t) -> RobotJointCommand:
        """Convert robot_joint_command LCM type to RobotJointCommand dataclass.

        Args:
            lcm_command: robot_joint_command LCM type

        Returns:
            RobotJointCommand dataclass
        """
        joint_positions = np.array(lcm_command.joint_positions)

        return RobotJointCommand(
            timestamp=lcm_command.timestamp / 1e9,  # Convert from nanoseconds
            joint_positions=joint_positions,
        )

    @staticmethod
    def robot_state_to_lcm(state: RobotState) -> robot_state_t:
        """Convert RobotState dataclass to robot_state_t LCM type.

        Args:
            state: RobotState dataclass with timestamp, joint_positions, joint_velocities, and motor_temperatures arrays

        Returns:
            robot_state_t LCM type ready for transmission
        """
        lcm_state = robot_state_t()
        lcm_state.timestamp = int(state.timestamp * 1e9)  # Convert to nanoseconds
        lcm_state.num_joints = len(state.joint_positions)
        lcm_state.joint_positions = state.joint_positions.tolist()
        lcm_state.joint_velocities = state.joint_velocities.tolist()
        lcm_state.motor_temperatures = state.motor_temperatures.tolist()

        return lcm_state

    @staticmethod
    def robot_state_from_lcm(lcm_state: robot_state_t) -> RobotState:
        """Convert robot_state_t LCM type to RobotState dataclass.

        Args:
            lcm_state: robot_state_t LCM type

        Returns:
            RobotState dataclass
        """
        joint_positions = np.array(lcm_state.joint_positions)
        joint_velocities = np.array(lcm_state.joint_velocities)
        motor_temperatures = np.array(lcm_state.motor_temperatures)

        return RobotState(
            timestamp=lcm_state.timestamp / 1e9,  # Convert from nanoseconds
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            motor_temperatures=motor_temperatures
        )

    @staticmethod
    def robot_tool_command_to_lcm(command: RobotToolCommand) -> robot_tool_command_t:
        """Convert RobotToolCommand dataclass to robot_tool_command_t LCM type.

        Args:
            command: RobotToolCommand dataclass with timestamp and pose (pin.SE3)

        Returns:
            robot_tool_command_t LCM type ready for transmission
        """
        lcm_command = robot_tool_command_t()
        lcm_command.timestamp = int(command.timestamp * 1e9)  # Convert to nanoseconds
        
        # Extract position from SE3
        lcm_command.position = command.pose.translation.tolist()
        
        # Extract quaternion from SE3 rotation matrix in wxyz format
        quat = pin.Quaternion(command.pose.rotation)
        # Quaternion format: [w, x, y, z]
        lcm_command.quaternion = [quat.w, quat.x, quat.y, quat.z]
        
        return lcm_command

    @staticmethod
    def robot_tool_command_from_lcm(lcm_command: robot_tool_command_t) -> RobotToolCommand:
        """Convert robot_tool_command_t LCM type to RobotToolCommand dataclass.

        Args:
            lcm_command: robot_tool_command_t LCM type

        Returns:
            RobotToolCommand dataclass
        """
        # Reconstruct position
        position = np.array(lcm_command.position)
        
        # Reconstruct rotation from quaternion in wxyz format [w, x, y, z]
        quat = pin.Quaternion(
            lcm_command.quaternion[0],  # w
            lcm_command.quaternion[1],  # x
            lcm_command.quaternion[2],  # y
            lcm_command.quaternion[3]   # z
        )
        rotation = quat.toRotationMatrix()
        
        # Create SE3 pose
        pose = pin.SE3(rotation, position)
        
        return RobotToolCommand(
            timestamp=lcm_command.timestamp / 1e9,  # Convert from nanoseconds
            pose=pose
        )
