"""Robot-level configuration for hardware device interfaces."""

from dataclasses import dataclass

from humanoid.types.actuator import ActuatorHardwareConfig


@dataclass
class RobotHardwareConfig:
    """Simulated or physical device interfaces attached to a robot."""

    actuators: ActuatorHardwareConfig | None = None
