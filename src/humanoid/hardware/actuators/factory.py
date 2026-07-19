"""Actuator-system construction from typed robot hardware configuration."""

from humanoid.hardware.actuators.config import ActuatorControlMode, ActuatorHardwareConfig
from humanoid.hardware.actuators.feetech.config import (
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.actuators.feetech.driver import FeetechActuatorDriver
from humanoid.hardware.actuators.simulation import SimulatedActuatorSystem
from humanoid.hardware.actuators.system import (
    ActuatorSystem,
    CompositeActuatorSystem,
)
from humanoid.types.process import Runtime


def create_actuator_system(
    runtime: Runtime,
    control_modes: dict[str, ActuatorControlMode],
    hardware: ActuatorHardwareConfig | None,
    initial_positions: dict[str, float],
) -> ActuatorSystem:
    """Create the simulation or real actuator system for one robot."""
    if runtime is Runtime.SIM:
        return SimulatedActuatorSystem(control_modes, initial_positions)

    if hardware is None:
        raise RuntimeError("Real actuator control requires configured actuator hardware.")
    if hardware.joints.keys() != control_modes.keys():
        raise ValueError("Real actuator bindings must match the supplied control modes.")
    feetech_controllers = {
        name: controller
        for name, controller in hardware.controllers.items()
        if isinstance(controller, FeetechActuatorControllerConfig)
    }
    if len(feetech_controllers) > 1:
        ports = [controller.port for controller in feetech_controllers.values()]
        if any(port is None for port in ports):
            raise ValueError(
                "Every Feetech controller must specify a port when multiple "
                "controllers are configured."
            )
        if len(set(ports)) != len(ports):
            raise ValueError("Feetech controllers must use distinct serial ports.")

    drivers = {}
    for controller_name, controller in hardware.controllers.items():
        controller_actuators = [
            actuator
            for actuator in hardware.joints.values()
            if actuator.controller == controller_name
        ]
        if isinstance(controller, FeetechActuatorControllerConfig):
            feetech_actuators = [
                actuator
                for actuator in controller_actuators
                if isinstance(actuator, FeetechActuatorConfig)
            ]
            if len(feetech_actuators) != len(controller_actuators):
                raise TypeError(f"Controller {controller_name!r} contains non-Feetech actuators.")
            control_modes_by_id = {
                actuator.actuator_id: control_modes[joint_name]
                for joint_name, actuator in hardware.joints.items()
                if actuator.controller == controller_name
            }
            drivers[controller_name] = FeetechActuatorDriver(
                feetech_actuators,
                control_modes_by_id,
                controller,
            )
            continue
        raise TypeError(f"Unsupported actuator controller: {type(controller).__name__}")

    return CompositeActuatorSystem(hardware.joints, control_modes, drivers)
