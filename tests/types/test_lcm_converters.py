import numpy as np
import pinocchio as pin

from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import RobotBaseCommand, RobotJointCommand, RobotState, RobotToolCommand


def test_robot_joint_command_conversion():
    """RobotJointCommand round-trips with num_positions and num_velocities stored separately."""
    # nq=9 (e.g. two continuous wheel joints), nv=7
    nq = 9
    nv = 7
    timestamp = 2.5

    q = np.random.default_rng(2).uniform(-1.0, 1.0, nq)
    v = np.random.default_rng(3).uniform(-0.5, 0.5, nv)

    cmd = RobotJointCommand(timestamp=timestamp, joint_positions=q, joint_velocities=v)
    lcm_cmd = LCMConverter.robot_joint_command_to_lcm(cmd)

    assert lcm_cmd.num_positions == nq
    assert lcm_cmd.num_velocities == nv
    np.testing.assert_allclose(lcm_cmd.joint_positions, q)
    np.testing.assert_allclose(lcm_cmd.joint_velocities, v)

    cmd_recovered = LCMConverter.robot_joint_command_from_lcm(lcm_cmd)

    assert np.isclose(cmd_recovered.timestamp, timestamp, rtol=1e-9)
    np.testing.assert_allclose(cmd_recovered.joint_positions, q)
    assert cmd_recovered.joint_velocities is not None
    np.testing.assert_allclose(cmd_recovered.joint_velocities, v)


def test_robot_joint_command_encode_decode():
    """RobotJointCommand survives a full binary encode/decode cycle with distinct sizes."""
    q = np.array([1.0, 0.0, 1.0, 2.0, 3.0])  # nq=5
    v = np.array([0.1, 0.2, 0.3, 0.4])  # nv=4

    cmd = RobotJointCommand(timestamp=0.1, joint_positions=q, joint_velocities=v)
    lcm_cmd = LCMConverter.robot_joint_command_to_lcm(cmd)

    recovered = LCMConverter.robot_joint_command_from_lcm(type(lcm_cmd).decode(lcm_cmd.encode()))

    np.testing.assert_allclose(recovered.joint_positions, q)
    assert recovered.joint_velocities is not None
    np.testing.assert_allclose(recovered.joint_velocities, v)


def test_robot_joint_command_none_velocities():
    """When joint_velocities is None, num_velocities=0 on the wire and decodes back to None."""
    q = np.array([0.1, 0.2, 0.3])
    cmd = RobotJointCommand(timestamp=0.0, joint_positions=q)
    lcm_cmd = LCMConverter.robot_joint_command_to_lcm(cmd)

    assert lcm_cmd.num_positions == len(q)
    assert lcm_cmd.num_velocities == 0
    assert lcm_cmd.joint_velocities == []

    recovered = LCMConverter.robot_joint_command_from_lcm(type(lcm_cmd).decode(lcm_cmd.encode()))
    assert recovered.joint_velocities is None


def test_robot_state_conversion():
    """RobotState round-trips with num_joints, num_positions, num_velocities stored separately."""
    # Simulate a robot where nq=9 (e.g. planar base adds 4, two continuous wheel joints
    # add 2 each, minus the 1 each they'd have as revolute = net +2, so 7 servos → nq=9),
    # nv=7, and 7 physical servo joints.
    n_joints = 7
    nq = 9
    nv = 7
    timestamp = 1.23456

    q = np.random.default_rng(0).uniform(-1.0, 1.0, nq)
    v = np.random.default_rng(1).uniform(-0.5, 0.5, nv)
    temps = np.arange(n_joints, dtype=float)

    state = RobotState(
        timestamp=timestamp, joint_positions=q, joint_velocities=v, actuator_temperatures=temps
    )

    lcm_state = LCMConverter.robot_state_to_lcm(state)

    assert lcm_state.num_joints == n_joints
    assert lcm_state.num_positions == nq
    assert lcm_state.num_velocities == nv
    assert lcm_state.timestamp == int(timestamp * 1e9)
    np.testing.assert_allclose(lcm_state.joint_positions, q)
    np.testing.assert_allclose(lcm_state.joint_velocities, v)
    np.testing.assert_allclose(lcm_state.actuator_temperatures, temps)

    state_recovered = LCMConverter.robot_state_from_lcm(lcm_state)

    assert np.isclose(state_recovered.timestamp, timestamp, rtol=1e-9)
    np.testing.assert_allclose(state_recovered.joint_positions, q)
    np.testing.assert_allclose(state_recovered.joint_velocities, v)
    np.testing.assert_allclose(state_recovered.actuator_temperatures, temps)


def test_robot_state_encode_decode():
    """RobotState survives a full binary encode/decode cycle with distinct sizes."""
    q = np.array([1.0, 0.0, 1.0, 2.0, 3.0])  # nq=5
    v = np.array([0.1, 0.2, 0.3, 0.4])  # nv=4
    temps = np.array([30.0, 31.0, 32.0])  # n_joints=3

    state = RobotState(
        timestamp=0.5, joint_positions=q, joint_velocities=v, actuator_temperatures=temps
    )
    lcm_state = LCMConverter.robot_state_to_lcm(state)

    recovered = LCMConverter.robot_state_from_lcm(type(lcm_state).decode(lcm_state.encode()))

    np.testing.assert_allclose(recovered.joint_positions, q)
    np.testing.assert_allclose(recovered.joint_velocities, v)
    np.testing.assert_allclose(recovered.actuator_temperatures, temps)


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
