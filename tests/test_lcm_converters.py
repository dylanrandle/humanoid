import numpy as np
import pinocchio as pin

from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.robot import RobotToolCommand


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
