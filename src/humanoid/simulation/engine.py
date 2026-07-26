"""Native single-world MuJoCo engine used by the LCM digital twin."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid.config.simulation import (
    DEFAULT_MUJOCO_SIMULATION_CONFIG,
    FLOOR_AND_CUBE_SCENE_CONFIG,
)
from humanoid.constants import DEFAULT_MUJOCO_SCENE
from humanoid.robots.base import Robot
from humanoid.robots.command import normalize_robot_joint_command
from humanoid.simulation.binding import (
    MujocoRobotBinding,
    resolve_mujoco_robot_binding,
)
from humanoid.simulation.scene import build_mujoco_scene
from humanoid.types.actuator import ActuatorControlMode
from humanoid.types.homing import HomingPreset
from humanoid.types.robot import (
    NormalizedRobotJointCommand,
    RobotConfig,
    RobotJointCommand,
    RobotState,
)
from humanoid.types.simulation import (
    FloorAndCubeSceneConfig,
    MujocoScene,
    MujocoSimulationConfig,
)

SIMULATED_ACTUATOR_TEMPERATURE = 25.0


class NativeMujocoEngine:
    """Own one compiled MuJoCo model and its deterministic physics state."""

    def __init__(
        self,
        robot_config: RobotConfig,
        config: MujocoSimulationConfig = DEFAULT_MUJOCO_SIMULATION_CONFIG,
        scene: MujocoScene = DEFAULT_MUJOCO_SCENE,
        floor_and_cube_config: FloorAndCubeSceneConfig = FLOOR_AND_CUBE_SCENE_CONFIG,
    ) -> None:
        self.config = config
        self.scene = scene
        self.robot = Robot(robot_config)
        self.spec = build_mujoco_scene(
            self.robot,
            scene,
            config,
            floor_and_cube_config,
        )
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.binding: MujocoRobotBinding = resolve_mujoco_robot_binding(
            self.model,
            self.robot,
        )
        self.reset()

    @property
    def physics_timestep(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        """Reset physics and actuator targets to the configured home state."""

        mujoco.mj_resetData(self.model, self.data)
        home = self.robot.config.homing_presets[HomingPreset.HOME]
        for joint in self.binding.joints:
            robot_joint_idx = self.robot.joint_name_to_idx(joint.name)
            position = self.robot.joint_position_from_q(home, robot_joint_idx)
            self.data.qpos[joint.qpos_address] = position
            self.data.ctrl[joint.actuator_id] = (
                position if joint.control_mode is ActuatorControlMode.POSITION else 0.0
            )

        root = self.binding.root
        root_q_slice = self.robot.get_root_q_slice()
        if root is not None and root_q_slice is not None:
            x, y, cos_yaw, sin_yaw = home[root_q_slice]
            yaw = math.atan2(float(sin_yaw), float(cos_yaw))
            self.data.qpos[list(root.qpos_addresses)] = [x, y, yaw]
            self.data.ctrl[list(root.actuator_ids)] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def apply_joint_command(self, command: RobotJointCommand) -> NormalizedRobotJointCommand:
        """Apply one full-stack command to named MuJoCo actuator targets."""

        normalized = normalize_robot_joint_command(
            command,
            self.robot.model.lowerPositionLimit,
            self.robot.model.upperPositionLimit,
            self.robot.model.velocityLimit,
        )

        for joint in self.binding.joints:
            robot_joint_idx = self.robot.joint_name_to_idx(joint.name)
            if joint.control_mode is ActuatorControlMode.POSITION:
                target = self.robot.joint_position_from_q(
                    normalized.joint_positions,
                    robot_joint_idx,
                )
            else:
                velocity_idx = self.robot.joint_idx_to_velocity_idx(robot_joint_idx)
                target = normalized.joint_velocities[velocity_idx]
            self.data.ctrl[joint.actuator_id] = float(target)

        root = self.binding.root
        root_v_slice = self.robot.get_root_v_slice()
        if root is not None and root_v_slice is not None:
            body_x, body_y, yaw_rate = normalized.joint_velocities[root_v_slice]
            yaw = float(self.data.qpos[root.qpos_addresses[2]])
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            self.data.ctrl[list(root.actuator_ids)] = [
                cos_yaw * body_x - sin_yaw * body_y,
                sin_yaw * body_x + cos_yaw * body_y,
                yaw_rate,
            ]
        return normalized

    def stop_velocity_actuators(self) -> None:
        """Stop the mobile base while position-controlled joints continue holding."""

        for joint in self.binding.joints:
            if joint.control_mode is ActuatorControlMode.VELOCITY:
                self.data.ctrl[joint.actuator_id] = 0.0
        if self.binding.root is not None:
            self.data.ctrl[list(self.binding.root.actuator_ids)] = 0.0

    def step(self, substeps: int = 1) -> None:
        if substeps <= 0:
            raise ValueError("MuJoCo substeps must be positive.")
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo produced non-finite robot state.")

    def read_robot_state(self, timestamp: float) -> RobotState:
        joint_idx_to_position: dict[int, float] = {}
        joint_idx_to_velocity: dict[int, float] = {}
        for joint in self.binding.joints:
            robot_joint_idx = self.robot.joint_name_to_idx(joint.name)
            joint_idx_to_position[robot_joint_idx] = float(self.data.qpos[joint.qpos_address])
            joint_idx_to_velocity[robot_joint_idx] = float(self.data.qvel[joint.qvel_address])

        q = self.robot.joint_positions_to_q(joint_idx_to_position)
        v = self.robot.joint_velocities_to_v(joint_idx_to_velocity)
        root = self.binding.root
        root_q_slice = self.robot.get_root_q_slice()
        root_v_slice = self.robot.get_root_v_slice()
        if root is not None and root_q_slice is not None and root_v_slice is not None:
            x, y, yaw = self.data.qpos[list(root.qpos_addresses)]
            q[root_q_slice] = [x, y, math.cos(float(yaw)), math.sin(float(yaw))]
            world_x, world_y, yaw_rate = self.data.qvel[list(root.qvel_addresses)]
            cos_yaw = math.cos(float(yaw))
            sin_yaw = math.sin(float(yaw))
            v[root_v_slice] = [
                cos_yaw * world_x + sin_yaw * world_y,
                -sin_yaw * world_x + cos_yaw * world_y,
                yaw_rate,
            ]

        return RobotState(
            timestamp=timestamp,
            joint_positions=q,
            joint_velocities=v,
            actuator_temperatures=np.full(
                len(self.binding.joints),
                SIMULATED_ACTUATOR_TEMPERATURE,
            ),
        )
