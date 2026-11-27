"""Track the latest state and command overlays received by a live viewer."""

import time

import numpy as np

from humanoid.config import ROBOT_CONFIG, ROBOT_CONFIGS
from humanoid.constants import EnvironmentVariable
from humanoid.types.actuator import ActuatorEffortSource
from humanoid.types.homing import HomingPreset
from humanoid.types.process import Runtime
from humanoid.types.robot import (
    RobotBaseCommand,
    RobotJointCommand,
    RobotName,
    RobotState,
    RobotToolCommand,
)
from humanoid.types.simulation import SimulationScene, SimulationState
from humanoid.visualizers.stream import Telemetry


def _hardware_robot(q: np.ndarray) -> RobotName:
    # Hardware feedback carries generalized coordinates, without a robot name.
    if ROBOT_CONFIG.homing_presets[HomingPreset.HOME].shape == q.shape:
        return ROBOT_CONFIG.name
    matches = [
        config.name
        for config in ROBOT_CONFIGS.values()
        if config.homing_presets[HomingPreset.HOME].shape == q.shape
    ]
    if len(matches) != 1:
        raise ValueError(
            f"No unique hardware model for {q.size} coordinates. Check {EnvironmentVariable.ROBOT}."
        )
    return matches[0]


class LiveTelemetry:
    def __init__(self) -> None:
        self.last_received = 0.0
        self.state: RobotState | SimulationState | None = None
        self.model_key: tuple[RobotName, Runtime, SimulationScene] | None = None
        self.joint_target: np.ndarray | None = None
        self.tool: RobotToolCommand | None = None
        self.base: RobotBaseCommand | None = None
        self.timestamp = 0.0

    def update(self, packets: list[Telemetry]) -> bool:
        if not packets:
            return False
        now = time.monotonic()
        if self.state is not None and now - self.last_received > 1.0:
            self.state = None
            self.joint_target = self.tool = self.base = None
        changed = False
        for packet in packets:
            if self._observe(packet, now):
                changed = True
        return changed

    def _observe(self, packet: Telemetry, now: float) -> bool:
        if isinstance(packet, RobotJointCommand):
            self.joint_target = packet.joint_positions
        elif isinstance(packet, RobotToolCommand):
            self.tool = packet
        elif isinstance(packet, RobotBaseCommand):
            self.base = packet
        else:
            if isinstance(packet, SimulationState):
                key = (packet.robot, Runtime.SIM, packet.scene)
                timestamp = packet.state[0]
            else:
                if packet.effort_source is ActuatorEffortSource.SIMULATION or (
                    packet.effort_source is None and isinstance(self.state, SimulationState)
                ):
                    return False  # SIMULATION/STATE also contains the moving scene.
                key = (_hardware_robot(packet.joint_positions), Runtime.REAL, SimulationScene.EMPTY)
                timestamp = packet.timestamp
            if self.state is not None and (key != self.model_key or timestamp < self.timestamp):
                self.joint_target = self.tool = self.base = None
            self.state, self.model_key = packet, key
            self.timestamp, self.last_received = timestamp, now
        return True
