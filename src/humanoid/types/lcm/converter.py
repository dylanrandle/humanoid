"""Converter between LCM types and Python dataclasses."""

from humanoid.types.lcm import motor_temperature_t
from humanoid.types.lcm.joint_position_t import joint_position_t
from humanoid.types.lcm.robot_command_t import robot_command_t
from humanoid.types.lcm.robot_state_t import robot_state_t
from humanoid.types.robot import RobotCommand, RobotState


class LCMConverter:
    """Handles conversion between LCM types and Python dataclasses."""

    @staticmethod
    def robot_command_to_lcm(command: RobotCommand) -> robot_command_t:
        """Convert RobotCommand dataclass to robot_command_t LCM type.

        Args:
            command: RobotCommand dataclass with timestamp and joint_positions dict

        Returns:
            robot_command_t LCM type ready for transmission
        """
        lcm_command = robot_command_t()
        lcm_command.timestamp = int(command.timestamp * 1e9)  # Convert to nanoseconds
        lcm_command.num_joints = len(command.joint_positions)
        lcm_command.joint_positions = []

        for name, position in command.joint_positions.items():
            joint_pos = joint_position_t()
            joint_pos.name = name
            joint_pos.position = position
            lcm_command.joint_positions.append(joint_pos)

        return lcm_command

    @staticmethod
    def robot_command_from_lcm(lcm_command: robot_command_t) -> RobotCommand:
        """Convert robot_command_t LCM type to RobotCommand dataclass.

        Args:
            lcm_command: robot_command_t LCM type

        Returns:
            RobotCommand dataclass
        """
        joint_positions = {
            jp.name: jp.position for jp in lcm_command.joint_positions
        }

        return RobotCommand(
            timestamp=lcm_command.timestamp / 1e9,  # Convert from nanoseconds
            joint_positions=joint_positions,
        )

    @staticmethod
    def robot_state_to_lcm(state: RobotState) -> robot_state_t:
        """Convert RobotState dataclass to robot_state_t LCM type.

        Args:
            state: RobotState dataclass with timestamp and joint_positions dict

        Returns:
            robot_state_t LCM type ready for transmission
        """
        lcm_state = robot_state_t()
        lcm_state.timestamp = int(state.timestamp * 1e9)  # Convert to nanoseconds
        lcm_state.num_joints = len(state.joint_positions)
        lcm_state.joint_positions = []
        lcm_state.motor_temperatures = []

        for name, position in state.joint_positions.items():
            joint_pos = joint_position_t()
            joint_pos.name = name
            joint_pos.position = position
            lcm_state.joint_positions.append(joint_pos)

        for name, temperature in state.motor_temperatures.items():
            motor_temp = motor_temperature_t()
            motor_temp.name = name
            motor_temp.temperature = temperature
            lcm_state.motor_temperatures.append(motor_temp)

        return lcm_state

    @staticmethod
    def robot_state_from_lcm(lcm_state: robot_state_t) -> RobotState:
        """Convert robot_state_t LCM type to RobotState dataclass.

        Args:
            lcm_state: robot_state_t LCM type

        Returns:
            RobotState dataclass
        """
        joint_positions = {
            jp.name: jp.position for jp in lcm_state.joint_positions
        }

        motor_temperatures = {
            mt.name: mt.temperature for mt in lcm_state.motor_temperatures
        }

        return RobotState(
            timestamp=lcm_state.timestamp / 1e9,  # Convert from nanoseconds
            joint_positions=joint_positions,
            motor_temperatures=motor_temperatures
        )
