"""Converter between LCM types and Python dataclasses."""

import math

import numpy as np
import pinocchio as pin

from humanoid.types.actuator import ActuatorHealth, ActuatorHealthReport
from humanoid.types.homing import HomingTarget
from humanoid.types.lcm import (
    actuator_health_report_t,
    homing_target_t,
    logging_status_t,
    node_rate_sample_t,
    orchestrator_event_t,
    orchestrator_mode_t,
    robot_base_command_t,
    robot_joint_command_t,
    robot_state_t,
    robot_tool_command_t,
)
from humanoid.types.logging import LoggingState, LoggingStatus
from humanoid.types.node import NodeRateSample
from humanoid.types.orchestrator import (
    EventKind,
    Mode,
    OrchestratorEvent,
    OrchestratorMode,
)
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotState,
    RobotToolCommand,
)


class LCMConverter:
    """Handles conversion between LCM types and Python dataclasses."""

    @staticmethod
    def actuator_health_report_to_lcm(
        report: ActuatorHealthReport,
    ) -> actuator_health_report_t:
        """Convert actuator health telemetry to its LCM representation."""
        lcm_report = actuator_health_report_t()
        lcm_report.timestamp = int(report.timestamp * 1e9)
        lcm_report.num_actuators = len(report.actuators)
        lcm_report.joint_names = [actuator.joint_name for actuator in report.actuators]
        lcm_report.controllers = [actuator.controller for actuator in report.actuators]
        lcm_report.actuator_ids = [actuator.actuator_id for actuator in report.actuators]
        lcm_report.healthy = [int(actuator.healthy) for actuator in report.actuators]
        lcm_report.temperatures_celsius = [
            actuator.temperature_celsius if actuator.temperature_celsius is not None else math.nan
            for actuator in report.actuators
        ]
        lcm_report.issues = [actuator.issue or "" for actuator in report.actuators]
        lcm_report.error = report.error or ""
        return lcm_report

    @staticmethod
    def actuator_health_report_from_lcm(
        lcm_report: actuator_health_report_t,
    ) -> ActuatorHealthReport:
        """Convert LCM actuator health telemetry to application types."""
        actuators = tuple(
            ActuatorHealth(
                joint_name=joint_name,
                controller=controller,
                actuator_id=actuator_id,
                healthy=bool(healthy),
                temperature_celsius=(None if math.isnan(temperature) else temperature),
                issue=issue or None,
            )
            for joint_name, controller, actuator_id, healthy, temperature, issue in zip(
                lcm_report.joint_names,
                lcm_report.controllers,
                lcm_report.actuator_ids,
                lcm_report.healthy,
                lcm_report.temperatures_celsius,
                lcm_report.issues,
                strict=True,
            )
        )
        return ActuatorHealthReport(
            timestamp=lcm_report.timestamp / 1e9,
            actuators=actuators,
            error=lcm_report.error or None,
        )

    @staticmethod
    def node_rate_sample_to_lcm(sample: NodeRateSample) -> node_rate_sample_t:
        """Convert NodeRateSample dataclass to node_rate_sample_t LCM type."""
        lcm_sample = node_rate_sample_t()
        lcm_sample.timestamp = int(sample.timestamp * 1e9)
        lcm_sample.node_name = sample.node_name
        lcm_sample.pid = sample.pid
        lcm_sample.target_rate_hz = sample.target_rate_hz
        lcm_sample.measured_rate_hz = sample.measured_rate_hz
        lcm_sample.cpu_percent = sample.cpu_percent
        lcm_sample.memory_rss_mb = sample.memory_rss_mb
        return lcm_sample

    @staticmethod
    def node_rate_sample_from_lcm(lcm_sample: node_rate_sample_t) -> NodeRateSample:
        """Convert node_rate_sample_t LCM type to NodeRateSample dataclass."""
        return NodeRateSample(
            timestamp=lcm_sample.timestamp / 1e9,
            node_name=lcm_sample.node_name,
            pid=lcm_sample.pid,
            target_rate_hz=lcm_sample.target_rate_hz,
            measured_rate_hz=lcm_sample.measured_rate_hz,
            cpu_percent=lcm_sample.cpu_percent,
            memory_rss_mb=lcm_sample.memory_rss_mb,
        )

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
        lcm_command.num_positions = len(command.joint_positions)
        lcm_command.joint_positions = command.joint_positions.tolist()
        lcm_command.joint_velocities = (
            command.joint_velocities.tolist() if command.joint_velocities is not None else []
        )
        lcm_command.num_velocities = len(lcm_command.joint_velocities)

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
        joint_velocities = (
            np.array(lcm_command.joint_velocities) if lcm_command.joint_velocities else None
        )

        return RobotJointCommand(
            timestamp=lcm_command.timestamp / 1e9,  # Convert from nanoseconds
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
        )

    @staticmethod
    def robot_state_to_lcm(state: RobotState) -> robot_state_t:
        """Convert RobotState dataclass to robot_state_t LCM type.

        Args:
            state: RobotState dataclass with timestamp, joint_positions, joint_velocities,
                   and actuator_temperatures arrays

        Returns:
            robot_state_t LCM type ready for transmission
        """
        lcm_state = robot_state_t()
        lcm_state.timestamp = int(state.timestamp * 1e9)  # Convert to nanoseconds
        lcm_state.num_joints = len(state.actuator_temperatures)
        lcm_state.num_positions = len(state.joint_positions)
        lcm_state.num_velocities = len(state.joint_velocities)
        lcm_state.joint_positions = state.joint_positions.tolist()
        lcm_state.joint_velocities = state.joint_velocities.tolist()
        lcm_state.actuator_temperatures = state.actuator_temperatures.tolist()

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
        actuator_temperatures = np.array(lcm_state.actuator_temperatures)

        return RobotState(
            timestamp=lcm_state.timestamp / 1e9,  # Convert from nanoseconds
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            actuator_temperatures=actuator_temperatures,
        )

    @staticmethod
    def robot_tool_command_to_lcm(command: RobotToolCommand) -> robot_tool_command_t:
        """Convert RobotToolCommand dataclass to robot_tool_command_t LCM type.

        Args:
            command: RobotToolCommand dataclass with timestamp, pose (pin.SE3),
                     and optional gripper_positions

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

        # Add gripper positions if provided
        if command.gripper_positions is not None:
            lcm_command.num_gripper_joints = len(command.gripper_positions)
            lcm_command.gripper_positions = command.gripper_positions.tolist()
        else:
            lcm_command.num_gripper_joints = 0
            lcm_command.gripper_positions = []

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
            lcm_command.quaternion[3],  # z
        )
        rotation = quat.toRotationMatrix()

        # Create SE3 pose
        pose = pin.SE3(rotation, position)

        # Extract gripper positions if present
        gripper_positions = None
        if lcm_command.num_gripper_joints > 0:
            gripper_positions = np.array(lcm_command.gripper_positions)

        return RobotToolCommand(
            timestamp=lcm_command.timestamp / 1e9,  # Convert from nanoseconds
            pose=pose,
            gripper_positions=gripper_positions,
        )

    @staticmethod
    def robot_base_command_to_lcm(command: RobotBaseCommand) -> robot_base_command_t:
        """Convert RobotBaseCommand dataclass to robot_base_command_t LCM type."""
        lcm_command = robot_base_command_t()
        lcm_command.timestamp = int(command.timestamp * 1e9)  # Convert to nanoseconds
        lcm_command.position = command.pose.translation.tolist()
        quat = pin.Quaternion(command.pose.rotation)
        lcm_command.quaternion = [quat.w, quat.x, quat.y, quat.z]
        return lcm_command

    @staticmethod
    def robot_base_command_from_lcm(lcm_command: robot_base_command_t) -> RobotBaseCommand:
        """Convert robot_base_command_t LCM type to RobotBaseCommand dataclass."""
        position = np.array(lcm_command.position)
        quat = pin.Quaternion(
            lcm_command.quaternion[0],  # w
            lcm_command.quaternion[1],  # x
            lcm_command.quaternion[2],  # y
            lcm_command.quaternion[3],  # z
        )
        pose = pin.SE3(quat.toRotationMatrix(), position)
        return RobotBaseCommand(
            timestamp=lcm_command.timestamp / 1e9,  # Convert from nanoseconds
            pose=pose,
        )

    @staticmethod
    def orchestrator_mode_to_lcm(msg: OrchestratorMode) -> orchestrator_mode_t:
        """Convert OrchestratorMode dataclass to orchestrator_mode_t LCM type."""
        lcm_msg = orchestrator_mode_t()
        lcm_msg.timestamp = int(msg.timestamp * 1e9)  # Convert to nanoseconds
        lcm_msg.mode = msg.mode.value
        return lcm_msg

    @staticmethod
    def orchestrator_mode_from_lcm(lcm_msg: orchestrator_mode_t) -> OrchestratorMode:
        """Convert orchestrator_mode_t LCM type to OrchestratorMode dataclass."""
        return OrchestratorMode(
            timestamp=lcm_msg.timestamp / 1e9,  # Convert from nanoseconds
            mode=Mode(lcm_msg.mode),
        )

    @staticmethod
    def orchestrator_event_to_lcm(msg: OrchestratorEvent) -> orchestrator_event_t:
        """Convert OrchestratorEvent dataclass to orchestrator_event_t LCM type."""
        lcm_msg = orchestrator_event_t()
        lcm_msg.timestamp = int(msg.timestamp * 1e9)  # Convert to nanoseconds
        lcm_msg.kind = msg.kind.value
        return lcm_msg

    @staticmethod
    def orchestrator_event_from_lcm(lcm_msg: orchestrator_event_t) -> OrchestratorEvent:
        """Convert orchestrator_event_t LCM type to OrchestratorEvent dataclass."""
        return OrchestratorEvent(
            timestamp=lcm_msg.timestamp / 1e9,  # Convert from nanoseconds
            kind=EventKind(lcm_msg.kind),
        )

    @staticmethod
    def homing_target_to_lcm(msg: HomingTarget) -> homing_target_t:
        """Convert HomingTarget dataclass to homing_target_t LCM type."""
        lcm_msg = homing_target_t()
        lcm_msg.timestamp = int(msg.timestamp * 1e9)  # Convert to nanoseconds
        lcm_msg.num_positions = len(msg.target_position)
        lcm_msg.target_position = msg.target_position.tolist()
        return lcm_msg

    @staticmethod
    def homing_target_from_lcm(lcm_msg: homing_target_t) -> HomingTarget:
        """Convert homing_target_t LCM type to HomingTarget dataclass."""
        return HomingTarget(
            timestamp=lcm_msg.timestamp / 1e9,  # Convert from nanoseconds
            target_position=np.array(lcm_msg.target_position),
        )

    @staticmethod
    def logging_status_to_lcm(status: LoggingStatus) -> logging_status_t:
        """Convert LoggingStatus dataclass to logging_status_t LCM type."""
        lcm_status = logging_status_t()
        lcm_status.timestamp = int(status.timestamp * 1e9)
        lcm_status.state = status.state.value
        lcm_status.file_name = status.file_name or ""
        lcm_status.error = status.error or ""
        return lcm_status

    @staticmethod
    def logging_status_from_lcm(lcm_status: logging_status_t) -> LoggingStatus:
        """Convert logging_status_t LCM type to LoggingStatus dataclass."""
        return LoggingStatus(
            timestamp=lcm_status.timestamp / 1e9,
            state=LoggingState(lcm_status.state),
            file_name=lcm_status.file_name or None,
            error=lcm_status.error or None,
        )
