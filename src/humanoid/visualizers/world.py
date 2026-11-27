"""Reconstruct scene poses and command overlays from the locally compiled robot model."""

import math
from dataclasses import replace

import mujoco
import numpy as np

from humanoid.config import DEFAULT_MUJOCO_SIMULATION_CONFIG, VISUALIZER_CONFIG
from humanoid.config.visualizer import (
    EMPTY_GRID_CELL_COLOR,
    EMPTY_GRID_CELL_SIZE,
    EMPTY_GRID_SECTION_COLOR,
    EMPTY_GRID_SECTION_SIZE,
    VIEWER_BACKGROUND,
    VIEWER_FILL_DIRECTION,
    VIEWER_FILL_INTENSITY,
    VIEWER_KEY_DIRECTION,
    VIEWER_KEY_INTENSITY,
    VIEWER_KEY_POSITION,
    VIEWER_LIGHT_AMBIENT,
)
from humanoid.robots.base import Robot
from humanoid.simulation.binding import resolve_mujoco_robot_binding
from humanoid.simulation.model import build_mujoco_spec, robot_floor_height
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotState, RobotToolCommand
from humanoid.types.simulation import SimulationScene, SimulationState

_URDF_VISUAL_GROUP = 1
_SIGNATURE = mujoco.mjtState.mjSTATE_INTEGRATION


def _style_scene(spec: mujoco.MjSpec) -> None:
    # Replace scene lighting so every viewer world uses the same light balance.
    for light in list(spec.lights):
        spec.delete(light)
    spec.visual.headlight.active = False
    # The lower fill and ambient approximation remain visible in shadow.
    spec.worldbody.add_light(
        name="reference_fill",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        dir=VIEWER_FILL_DIRECTION,
        castshadow=False,
        ambient=VIEWER_LIGHT_AMBIENT,
        diffuse=[VIEWER_FILL_INTENSITY] * 3,
        specular=[0, 0, 0],
    )
    spec.worldbody.add_light(
        name="reference_light",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=VIEWER_KEY_POSITION,
        dir=VIEWER_KEY_DIRECTION,
        ambient=[0, 0, 0],
        diffuse=[VIEWER_KEY_INTENSITY] * 3,
    )
    spec.add_texture(
        name="reference_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=VIEWER_BACKGROUND,
        rgb2=VIEWER_BACKGROUND,
        width=256,
        height=1536,
    )


def _style_reference_floor(spec: mujoco.MjSpec) -> None:
    # One texture tile spans a section; the tile edges are the major grid lines.
    resolution = 512
    pixels = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    spacing = round(resolution * EMPTY_GRID_CELL_SIZE / EMPTY_GRID_SECTION_SIZE)
    pixels[::spacing, :] = EMPTY_GRID_CELL_COLOR
    pixels[:, ::spacing] = EMPTY_GRID_CELL_COLOR
    pixels[0, :] = EMPTY_GRID_SECTION_COLOR
    pixels[:, 0] = EMPTY_GRID_SECTION_COLOR
    spec.add_texture(
        name="reference_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=resolution,
        height=resolution,
        nchannel=3,
        data=pixels.ravel(),
    )
    material = spec.add_material(
        name="reference_ground",
        texuniform=True,
        texrepeat=[1 / EMPTY_GRID_SECTION_SIZE] * 2,
        specular=0,
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "reference_grid"
    floor = spec.geom("reference_floor")
    floor.material = material.name
    floor.rgba = [1, 1, 1, 1]


class ViewerWorld:
    def __init__(self, robot: Robot, scene: SimulationScene, runtime: Runtime):
        self.robot = robot
        spec = build_mujoco_spec(robot, replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=scene))
        if runtime is Runtime.REAL or scene is SimulationScene.EMPTY:
            spec.worldbody.add_geom(
                name="reference_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[2, 2, 0.01],
                pos=[0, 0, robot_floor_height(robot)],
                rgba=[0.4, 0.42, 0.44, 1],
                contype=0,
                conaffinity=0,
            )
        _style_scene(spec)
        if scene is SimulationScene.EMPTY:
            _style_reference_floor(spec)
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
