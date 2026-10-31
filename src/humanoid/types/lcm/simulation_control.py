"""Wire conversions for scene reset requests and simulator observations."""

import numpy as np

from humanoid.types.lcm import (
    simulation_reset_status_t,
    simulation_reset_t,
    simulation_state_t,
)
from humanoid.types.lcm.converter import LCMConverter
from humanoid.types.simulation import (
    SimulationReset,
    SimulationResetStatus,
    SimulationState,
)


def reset_to_lcm(value: SimulationReset) -> simulation_reset_t:
    message = simulation_reset_t()
    message.timestamp = round(value.timestamp * 1e9)
    message.request_id, message.task, message.seed = value.request_id, value.task, value.seed
    return message


def reset_from_lcm(message: simulation_reset_t) -> SimulationReset:
    return SimulationReset(message.timestamp / 1e9, message.request_id, message.task, message.seed)


def reset_status_to_lcm(value: SimulationResetStatus) -> simulation_reset_status_t:
    message = simulation_reset_status_t()
    message.timestamp = round(value.timestamp * 1e9)
    message.request_id, message.error = value.request_id, value.error
    return message


def reset_status_from_lcm(message: simulation_reset_status_t) -> SimulationResetStatus:
    return SimulationResetStatus(message.timestamp / 1e9, message.request_id, message.error)


def state_to_lcm(value: SimulationState) -> simulation_state_t:
    message = simulation_state_t()
    message.timestamp = round(value.timestamp * 1e9)
    message.reset_id, message.sim_seconds = value.reset_id, value.sim_seconds
    message.nq, message.qpos = len(value.qpos), value.qpos.tolist()
    message.nv, message.qvel = len(value.qvel), value.qvel.tolist()
    message.state = LCMConverter.robot_state_to_lcm(value.state)
    message.command = LCMConverter.robot_joint_command_to_lcm(value.command)
    message.error = value.error
    return message


def state_from_lcm(message: simulation_state_t) -> SimulationState:
    return SimulationState(
        message.timestamp / 1e9,
        message.reset_id,
        message.sim_seconds,
        np.asarray(message.qpos),
        np.asarray(message.qvel),
        LCMConverter.robot_state_from_lcm(message.state),
        LCMConverter.robot_joint_command_from_lcm(message.command),
        message.error,
    )
