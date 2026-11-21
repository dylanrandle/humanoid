"""Reconstruct a read-only view of a running simulation from its LCM frames."""

from dataclasses import replace

import mujoco

from humanoid.config import DEFAULT_MUJOCO_SIMULATION_CONFIG, ROBOT_CONFIGS
from humanoid.robots.base import Robot
from humanoid.simulation.model import build_mujoco_spec
from humanoid.types.simulation import SimulationState


class SimulationMirror:
    def __init__(self, frame: SimulationState):
        self.robot = frame.robot
        self.scene = frame.scene
        config = replace(DEFAULT_MUJOCO_SIMULATION_CONFIG, scene=frame.scene)
        self.model = build_mujoco_spec(Robot(ROBOT_CONFIGS[frame.robot]), config).compile()
        self.data = mujoco.MjData(self.model)
        self.apply(frame)

    def matches(self, frame: SimulationState) -> bool:
        return self.robot == frame.robot and self.scene == frame.scene

    def apply(self, frame: SimulationState) -> None:
        signature = mujoco.mjtState.mjSTATE_INTEGRATION
        if not self.matches(frame) or frame.state.size != mujoco.mj_stateSize(
            self.model, signature
        ):
            raise ValueError("Simulation frame does not match the viewer model.")
        mujoco.mj_setState(self.model, self.data, frame.state, signature)
        mujoco.mj_forward(self.model, self.data)
