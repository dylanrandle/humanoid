"""Contact-physics regressions for a full-height lift, hold, and base turn."""

import mujoco
import numpy as np
import pinocchio as pin
import pytest

from humanoid.simulation.tasks.cleanup.kinematics import ArmKinematics
from humanoid.simulation.tasks.cleanup.motion import grasp_rotation
from humanoid.simulation.tasks.cleanup.task import CleanupTask
from humanoid.types.robot import RobotJointCommand

MAX_GRASP_SLIP = 0.001  # 1 mm, including contact settling.


@pytest.mark.parametrize("name", ["orange_car", "red_block", "yellow_block", "green_block"])
@pytest.mark.parametrize("heading", [0, 1])
def test_object_stays_in_the_fingers_during_lift_and_turn(name, heading):
    task = CleanupTask()
    engine = task.create_engine(0)
    robot = engine.robot
    kinematics = ArmKinematics(robot, task.arm_config)

    def advance(seconds):
        engine.step(round(seconds / engine.physics_timestep))

    advance(0.4)
    item = task._item(engine, name)
    yaw = np.arctan2(item.pose.rotation[1, 0], item.pose.rotation[0, 0]) + heading * np.pi / 2
    base = item.pose.translation[:2] - np.array([-np.sin(yaw), np.cos(yaw)]) * 0.305
    q = engine.read_robot_state(0).joint_positions
    root_q = robot.get_root_q_slice()
    root = engine.binding.root
    assert root_q is not None and root is not None
    q[root_q] = [*base, np.cos(yaw), np.sin(yaw)]
    q[kinematics.indices] = task.arm_config.grasp_seed_arm_positions
    grasp = pin.SE3(
        grasp_rotation(yaw), np.r_[item.pose.translation[:2], task.scene.floor_height + 0.017]
    )
    q = kinematics.solve(grasp, q)
    # Initialize a controlled grasp fixture. From here only actuator/contact
    # physics moves the robot and toy; full-stack approach tests run separately.
    for joint in engine.binding.joints:
        engine.data.qpos[joint.qpos_address] = robot.joint_position_from_q(
            q, robot.joint_name_to_idx(joint.name)
        )
    engine.data.qpos[list(root.qpos_addresses)] = [*base, yaw]
    mujoco.mj_forward(engine.model, engine.data)

    def command():
        engine.apply_joint_command(RobotJointCommand(0, q, np.zeros(robot.model.nv)))

    def held_position():
        tool = kinematics.grasp_pose(engine.read_robot_state(0).joint_positions)
        return tool.actInv(task._item(engine, name).pose.translation)

    command()
    advance(1)
    robot.set_gripper_positions(q, np.array([task.arm_config.closed_position]))
    command()
    advance(2)
    initial = held_position()
    for fraction in np.linspace(0, 1, 120):
        target = grasp.copy()
        target.translation[2] += (
            (task.arm_config.transit_height - 0.017) * fraction**2 * (3 - 2 * fraction)
        )
        q = kinematics.solve(target, q)
        command()
        advance(0.02)
    advance(3)
    assert np.linalg.norm(held_position() - initial) < MAX_GRASP_SLIP
    # Rotate the base with the object supported only by physical finger contacts.
    velocity = np.zeros(robot.model.nv)
    root_v = robot.get_root_v_slice()
    assert root_v is not None
    velocity[root_v] = [0, 0, 0.2]
    engine.apply_joint_command(RobotJointCommand(0, q, velocity))
    advance(1.5)
    command()
    advance(1)
    assert np.linalg.norm(held_position() - initial) < MAX_GRASP_SLIP
    assert task._item(engine, name).corners[:, 2].min() > task.scene.floor_height + 0.10
    assert not any(warning.number for warning in engine.data.warning)
