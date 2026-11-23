"""Reconstruct scene poses and command overlays from the locally compiled robot model."""

import math
from dataclasses import replace

import mujoco
import numpy as np

from humanoid.config import DEFAULT_MUJOCO_SIMULATION_CONFIG, VISUALIZER_CONFIG
from humanoid.robots.base import Robot
from humanoid.simulation.binding import resolve_mujoco_robot_binding
from humanoid.simulation.model import build_mujoco_spec
from humanoid.types.homing import HomingPreset
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotState, RobotToolCommand
from humanoid.types.simulation import SimulationScene, SimulationState

_URDF_VISUAL_GROUP = 1
_SIGNATURE = mujoco.mjtState.mjSTATE_INTEGRATION


class ViewerWorld:
    def __init__(self, robot: Robot, scene: SimulationScene, runtime: Runtime):
        self.robot = robot
        spec = build_mujoco_spec(robot, replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=scene))
        if runtime is Runtime.REAL:
            floor_z = 0.0
            if robot.config.wheels:
                home = robot.config.homing_presets[HomingPreset.HOME]
                floor_z = robot.get_frame_pose(
                    robot.config.wheels[0].floor_frame, home
                ).translation[2]
            spec.worldbody.add_geom(
                name="reference_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[2, 2, 0.01],
                pos=[0, 0, floor_z],
                rgba=[0.4, 0.42, 0.44, 1],
                contype=0,
                conaffinity=0,
            )
        self.model = model = spec.compile()
        self.data = mujoco.MjData(model)
        self.joint_ghost = mujoco.MjData(model)
        self.tool_ghost = mujoco.MjData(model)
        self.show_joints = self.show_tool = False
        self.joint_geoms = np.flatnonzero(model.geom_group == _URDF_VISUAL_GROUP)
        tool_joint = robot.model.frames[robot.get_frame_id(robot.config.tool.frame)].parentJoint
        subtree = set(robot.model.subtrees[tool_joint])
        self.tool_geoms = np.array(
            [
                i
                for i in self.joint_geoms
                if (link := model.body(model.geom_bodyid[i]).name)
                and robot.model.existFrame(link)
                and robot.model.frames[robot.get_frame_id(link)].parentJoint in subtree
            ],
            dtype=int,
        )
        self.binding = resolve_mujoco_robot_binding(model, robot)
        self.joints = [
            (joint.qpos_address, robot.joint_name_to_idx(joint.name))
            for joint in self.binding.joints
        ]
        self.mimics = [
            (
                model.jnt_qposadr[model.eq_obj1id[i]],
                model.jnt_qposadr[model.eq_obj2id[i]],
                model.eq_data[i, :2],
            )
            for i in range(model.neq)
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[i] >= 0
        ]

    def _pose(self, data: mujoco.MjData, q: np.ndarray, *, at_measured_base: bool = False) -> None:
        if q.shape != (self.robot.model.nq,) or not np.isfinite(q).all():
            raise ValueError(f"Configuration does not match {self.robot.config.name}.")
        for address, index in self.joints:
            data.qpos[address] = self.robot.joint_position_from_q(q, index)
        root = self.binding.root
        root_slice = self.robot.get_root_q_slice()
        if root is not None and root_slice is not None:
            addresses = list(root.qpos_addresses)
            if at_measured_base:
                data.qpos[addresses] = self.data.qpos[addresses]
            else:
                x, y, c, s = q[root_slice]
                data.qpos[addresses] = [x, y, math.atan2(s, c)]
        # Kinematic playback does not step the solver that drives mimic joints in simulation.
        for target, source, (offset, scale) in self.mimics:
            data.qpos[target] = offset + scale * data.qpos[source]
        mujoco.mj_kinematics(self.model, data)

    def apply(
        self,
        state: SimulationState | RobotState,
        joint_target: np.ndarray | None,
        tool: RobotToolCommand | None,
    ) -> None:
        if isinstance(state, SimulationState):
            if state.state.size != mujoco.mj_stateSize(self.model, _SIGNATURE):
                raise ValueError(
                    "Simulation state size differs. Check both deployments and assets."
                )
            mujoco.mj_setState(self.model, self.data, state.state, _SIGNATURE)
        else:
            self._pose(self.data, state.joint_positions)
        mujoco.mj_forward(self.model, self.data)
        self.show_joints = (
            joint_target is not None and VISUALIZER_CONFIG.show_commanded_joint_positions
        )
        self.show_tool = tool is not None and VISUALIZER_CONFIG.show_commanded_tool_pose
        if joint_target is not None and self.show_joints:
            self._pose(self.joint_ghost, joint_target, at_measured_base=True)
        if tool is not None and self.show_tool:
            if not np.isfinite(tool.pose.homogeneous).all():
                raise ValueError("Invalid tool command pose.")
            target = tool.pose
            q = (
                state.joint_positions.copy()
                if isinstance(state, RobotState)
                else self._simulated_configuration()
            )
            base = self.robot.get_base_pose(q)
            if base is not None:
                target = base * target
            if tool.gripper_positions is not None:
                indices = self.robot.get_gripper_position_indices()
                if (
                    tool.gripper_positions.shape != (len(indices),)
                    or not np.isfinite(tool.gripper_positions).all()
                ):
                    raise ValueError("Invalid commanded gripper positions.")
                q[indices] = np.clip(
                    tool.gripper_positions,
                    self.robot.model.lowerPositionLimit[indices],
                    self.robot.model.upperPositionLimit[indices],
                )
            self._pose(self.tool_ghost, q)
            offset = target * self.robot.get_tool_pose(q).inverse()
            positions = self.tool_ghost.geom_xpos[self.tool_geoms]
            rotations = self.tool_ghost.geom_xmat[self.tool_geoms].reshape(-1, 3, 3)
            self.tool_ghost.geom_xpos[self.tool_geoms] = (
                positions @ offset.rotation.T + offset.translation
            )
            self.tool_ghost.geom_xmat[self.tool_geoms] = (offset.rotation @ rotations).reshape(
                -1, 9
            )

    def _simulated_configuration(self) -> np.ndarray:
        q = self.robot.joint_positions_to_q(
            {index: float(self.data.qpos[address]) for address, index in self.joints}
        )
        root = self.binding.root
        root_slice = self.robot.get_root_q_slice()
        if root is not None and root_slice is not None:
            x, y, yaw = self.data.qpos[list(root.qpos_addresses)]
            q[root_slice] = [x, y, math.cos(yaw), math.sin(yaw)]
        return q
