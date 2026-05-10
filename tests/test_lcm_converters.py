import numpy as np
import pinocchio as pin

from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import RobotBaseCommand, RobotToolCommand


def test_robot_tool_command_conversion():
    """Test that position and quaternion are recovered correctly through LCM conversion."""
    # Create a non-trivial position
    position = np.array([1.5, -2.3, 4.7])

    # Create a non-trivial rotation (45 degrees around arbitrary axis)
    angle = np.pi / 4
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)  # normalize
    rotation = pin.AngleAxis(angle, axis).toRotationMatrix()

    # Create SE3 pose
    pose = pin.SE3(rotation, position)

    # Create RobotToolCommand with non-trivial timestamp
    timestamp = 123.456789
    cmd = RobotToolCommand(timestamp=timestamp, pose=pose)

    # Convert to LCM
    lcm_cmd = LCMConverter.robot_tool_command_to_lcm(cmd)

    # Assert LCM message has correct timestamp (in nanoseconds)
    expected_timestamp_ns = int(timestamp * 1e9)
    assert lcm_cmd.timestamp == expected_timestamp_ns

    # Assert LCM message has correct position
    np.testing.assert_allclose(lcm_cmd.position, position, atol=1e-9)

    # Assert LCM message has correct quaternion (wxyz format)
    quat_expected = pin.Quaternion(rotation)
    np.testing.assert_allclose(
        lcm_cmd.quaternion,
        [quat_expected.w, quat_expected.x, quat_expected.y, quat_expected.z],
        atol=1e-9,
    )

    # Assert quaternion is normalized
    quat_norm = np.linalg.norm(lcm_cmd.quaternion)
    assert np.isclose(quat_norm, 1.0, atol=1e-9)

    # Convert back from LCM
    cmd_recovered = LCMConverter.robot_tool_command_from_lcm(lcm_cmd)

    # Assert timestamp is recovered correctly
    assert np.isclose(cmd_recovered.timestamp, timestamp, rtol=1e-9)

    # Assert position is recovered correctly
    np.testing.assert_allclose(cmd_recovered.pose.translation, position, atol=1e-9)

    # Assert rotation is recovered correctly
    np.testing.assert_allclose(cmd_recovered.pose.rotation, rotation, atol=1e-9)


def test_robot_tool_command_with_gripper():
    """Test RobotToolCommand conversion with gripper positions."""
    position = np.array([0.5, 0.3, 0.8])
    rotation = pin.rpy.rpyToMatrix(0.1, 0.2, 0.3)
    pose = pin.SE3(rotation, position)
    gripper_positions = np.array([0.01, 0.02])
    timestamp = 42.123

    cmd = RobotToolCommand(timestamp=timestamp, pose=pose, gripper_positions=gripper_positions)

    # Convert to LCM
    lcm_cmd = LCMConverter.robot_tool_command_to_lcm(cmd)

    # Check gripper data
    assert lcm_cmd.num_gripper_joints == len(gripper_positions)
    np.testing.assert_allclose(lcm_cmd.gripper_positions, gripper_positions, atol=1e-9)

    # Convert back
    cmd_recovered = LCMConverter.robot_tool_command_from_lcm(lcm_cmd)

    # Verify gripper positions are recovered
    assert cmd_recovered.gripper_positions is not None
    np.testing.assert_allclose(cmd_recovered.gripper_positions, gripper_positions, atol=1e-9)


def test_robot_base_command_conversion():
    """Test that base position and quaternion are recovered correctly through LCM conversion."""
    position = np.array([0.7, -1.2, 0.0])

    angle = np.pi / 3
    axis = np.array([0.0, 0.0, 1.0])
    rotation = pin.AngleAxis(angle, axis).toRotationMatrix()

    pose = pin.SE3(rotation, position)
    timestamp = 987.654321
    cmd = RobotBaseCommand(timestamp=timestamp, pose=pose)

    lcm_cmd = LCMConverter.robot_base_command_to_lcm(cmd)

    expected_timestamp_ns = int(timestamp * 1e9)
    assert lcm_cmd.timestamp == expected_timestamp_ns

    np.testing.assert_allclose(lcm_cmd.position, position, atol=1e-9)

    quat_expected = pin.Quaternion(rotation)
    np.testing.assert_allclose(
        lcm_cmd.quaternion,
        [quat_expected.w, quat_expected.x, quat_expected.y, quat_expected.z],
        atol=1e-9,
    )

    quat_norm = np.linalg.norm(lcm_cmd.quaternion)
    assert np.isclose(quat_norm, 1.0, atol=1e-9)

    cmd_recovered = LCMConverter.robot_base_command_from_lcm(lcm_cmd)

    assert np.isclose(cmd_recovered.timestamp, timestamp, rtol=1e-9)
    np.testing.assert_allclose(cmd_recovered.pose.translation, position, atol=1e-9)
    np.testing.assert_allclose(cmd_recovered.pose.rotation, rotation, atol=1e-9)


def test_robot_base_command_identity():
    """Test that an identity SE3 round-trips cleanly."""
    cmd = RobotBaseCommand(timestamp=0.0, pose=pin.SE3.Identity())

    lcm_cmd = LCMConverter.robot_base_command_to_lcm(cmd)
    cmd_recovered = LCMConverter.robot_base_command_from_lcm(lcm_cmd)

    np.testing.assert_allclose(cmd_recovered.pose.translation, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(cmd_recovered.pose.rotation, np.eye(3), atol=1e-9)
    assert cmd_recovered.timestamp == 0.0
