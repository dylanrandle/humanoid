"""Feetech actuator driver and unit conversions."""

import math
from collections import defaultdict
from typing import Any, cast

import scservo_sdk as scs
from vassar_feetech_servo_sdk import ServoController

from humanoid.hardware.actuators.driver import ActuatorDriver
from humanoid.hardware.actuators.feetech.config import (
    FEETECH_ACTUATOR_ID_MAX,
    FEETECH_ACTUATOR_ID_MIN,
    FEETECH_SPEED_UNIT_RAD_S,
    FeetechActuatorConfig,
    FeetechActuatorControllerConfig,
    FeetechPIDGains,
    validate_feetech_acceleration,
)
from humanoid.logger import get_logger
from humanoid.types.actuator import ActuatorControlMode

logger = get_logger(__name__)

POS_MIN = 0
POS_MAX = 4095
POS_MID = (POS_MAX + POS_MIN) / 2
ADDR_TEMPERATURE = 63
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_SPEED = 58
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_OPERATING_MODE = 33
ADDR_TORQUE_ENABLE = 40
ADDR_P_GAIN = 21
ADDR_D_GAIN = 22
ADDR_I_GAIN = 23
POSITION_PID_REGISTERS = (
    ("p", ADDR_P_GAIN),
    ("i", ADDR_I_GAIN),
    ("d", ADDR_D_GAIN),
)
OPERATING_MODE = {
    ActuatorControlMode.POSITION: 0,
    ActuatorControlMode.VELOCITY: 1,
}
# BIT15 encodes direction in the raw speed register.
SPEED_UNIT_RAD_S = FEETECH_SPEED_UNIT_RAD_S
DEFAULT_POSITION_SPEED = 32767
MIN_POSITION_SPEED = 1
HLS_FULL_TORQUE = 1000
UTILITY_CONTROLLER = "utility"
POSITION_DATA_LENGTH = 2
SPEED_DATA_LENGTH = 2
TEMPERATURE_DATA_LENGTH = 1
PRESENT_FEEDBACK_DATA_LENGTH = 8


class FeetechActuatorDriver(ServoController, ActuatorDriver):
    """Drive a configured group of Feetech actuators on one controller."""

    def __init__(
        self,
        actuator_configs: list[FeetechActuatorConfig],
        control_modes: dict[int, ActuatorControlMode],
        controller_config: FeetechActuatorControllerConfig,
        *,
        configure_on_connect: bool = True,
    ):
        self.actuator_configs = {actuator.actuator_id: actuator for actuator in actuator_configs}
        if len(self.actuator_configs) != len(actuator_configs):
            raise ValueError("Feetech actuator IDs must be unique within one controller.")
        actuator_ids = list(self.actuator_configs)
        if control_modes.keys() != self.actuator_configs.keys():
            raise ValueError("Feetech control modes must match the configured actuator IDs.")
        invalid_pid_ids: list[int] = [
            actuator_id
            for actuator_id, control_mode in control_modes.items()
            if control_mode is not ActuatorControlMode.POSITION
            and self.actuator_configs[actuator_id].position_pid is not None
        ]
        if invalid_pid_ids:
            raise ValueError(
                "Feetech position PID gains require position control; invalid actuator IDs: "
                + ", ".join(str(actuator_id) for actuator_id in invalid_pid_ids)
            )
        self.control_modes = dict(control_modes)
        self._configure_on_connect = configure_on_connect
        self._last_positions: dict[int, float] = {}
        self._health_issues: dict[int, str] = {}
        ServoController.__init__(
            self,
            servo_ids=actuator_ids,
            servo_type=controller_config.servo_type.value,
            port=controller_config.port,
            baudrate=controller_config.baud_rate,
        )
        ActuatorDriver.__init__(self, actuator_ids=actuator_ids)
        # Keep the SDK and hardware interface on one mutable identity list.
        self.servo_ids = self.actuator_ids

    @classmethod
    def for_actuator_ids(
        cls,
        actuator_ids: list[int],
        control_mode: ActuatorControlMode = ActuatorControlMode.POSITION,
        controller_config: FeetechActuatorControllerConfig | None = None,
        *,
        configure_on_connect: bool = False,
    ) -> "FeetechActuatorDriver":
        """Build an ad-hoc driver for Feetech maintenance utilities."""
        actuator_configs = [
            FeetechActuatorConfig(
                controller=UTILITY_CONTROLLER,
                actuator_id=actuator_id,
            )
            for actuator_id in actuator_ids
        ]
        return cls(
            actuator_configs,
            dict.fromkeys(actuator_ids, control_mode),
            controller_config or FeetechActuatorControllerConfig(),
            configure_on_connect=configure_on_connect,
        )

    def connect(self) -> None:
        """Connect each actuator with a safe goal before torque is enabled."""
        self._last_positions.clear()
        self._health_issues.clear()
        super().connect()
        if not self._configure_on_connect:
            return
        try:
            for actuator_id, control_mode in self.control_modes.items():
                self._configure_actuator(actuator_id, control_mode)
        except Exception:
            super().disconnect()
            raise

    def _configure_actuator(
        self,
        actuator_id: int,
        control_mode: ActuatorControlMode,
    ) -> None:
        assert self.packet_handler, "Not connected"
        self._set_torque_enabled(actuator_id, enabled=False)

        expected_mode = OPERATING_MODE[control_mode]
        mode, result, error = self.packet_handler.read1ByteTxRx(
            actuator_id,
            ADDR_OPERATING_MODE,
        )
        if result != 0 or error != 0:
            raise RuntimeError(f"Failed to read operating mode for actuator {actuator_id}.")
        if mode != expected_mode:
            if not self.set_operating_mode(actuator_id, expected_mode):
                raise RuntimeError(f"Failed to set operating mode for actuator {actuator_id}.")
            mode, result, error = self.packet_handler.read1ByteTxRx(
                actuator_id,
                ADDR_OPERATING_MODE,
            )
            if result != 0 or error != 0 or mode != expected_mode:
                raise RuntimeError(
                    f"Operating mode verification failed for actuator {actuator_id}."
                )

        position_pid = self.actuator_configs[actuator_id].position_pid
        if position_pid is not None:
            self._configure_position_pid(actuator_id, position_pid)

        self._stage_safe_goal(actuator_id, control_mode)
        self._set_torque_enabled(actuator_id, enabled=True)

    def _configure_position_pid(
        self,
        actuator_id: int,
        gains: FeetechPIDGains,
    ) -> None:
        """Write changed EEPROM gains and verify the persistent values."""
        assert self.packet_handler, "Not connected"
        current = self._read_position_pid(actuator_id)
        if current == gains:
            return

        result, error = self.packet_handler.unLockEprom(actuator_id)
        if result != 0 or error != 0:
            raise RuntimeError(f"Failed to unlock EEPROM for actuator {actuator_id}.")

        write_error: Exception | None = None
        try:
            for gain_name, address in POSITION_PID_REGISTERS:
                expected = getattr(gains, gain_name)
                if getattr(current, gain_name) == expected:
                    continue
                result, error = self.packet_handler.write1ByteTxRx(
                    actuator_id,
                    address,
                    expected,
                )
                if result != 0 or error != 0:
                    raise RuntimeError(
                        f"Failed to write {gain_name.upper()} gain for actuator {actuator_id}."
                    )
        except Exception as exc:
            write_error = exc

        result, error = self.packet_handler.LockEprom(actuator_id)
        if result != 0 or error != 0:
            lock_error = RuntimeError(f"Failed to lock EEPROM for actuator {actuator_id}.")
            if write_error is not None:
                raise ExceptionGroup(
                    f"Failed to configure position PID for actuator {actuator_id}.",
                    [write_error, lock_error],
                )
            raise lock_error
        if write_error is not None:
            raise write_error

        actual = self._read_position_pid(actuator_id)
        if actual != gains:
            raise RuntimeError(
                f"Position PID verification failed for actuator {actuator_id}: "
                f"expected {gains}, read {actual}."
            )
        logger.info(
            "Configured position PID for actuator %s: P=%s I=%s D=%s",
            actuator_id,
            gains.p,
            gains.i,
            gains.d,
        )

    def _read_position_pid(self, actuator_id: int) -> FeetechPIDGains:
        assert self.packet_handler, "Not connected"
        values: dict[str, int] = {}
        for gain_name, address in POSITION_PID_REGISTERS:
            value, result, error = self.packet_handler.read1ByteTxRx(actuator_id, address)
            if result != 0 or error != 0:
                raise RuntimeError(
                    f"Failed to read {gain_name.upper()} gain for actuator {actuator_id}."
                )
            values[gain_name] = value
        return FeetechPIDGains(**values)

    def _stage_safe_goal(
        self,
        actuator_id: int,
        control_mode: ActuatorControlMode,
    ) -> None:
        assert self.packet_handler, "Not connected"
        if control_mode is ActuatorControlMode.VELOCITY:
            address = ADDR_GOAL_SPEED
            value = 0
        else:
            address = ADDR_GOAL_POSITION
            value = super().read_position(actuator_id)
        result, error = self.packet_handler.write2ByteTxRx(actuator_id, address, value)
        if result != 0 or error != 0:
            raise RuntimeError(f"Failed to stage a safe goal for actuator {actuator_id}.")

    def _set_torque_enabled(self, actuator_id: int, *, enabled: bool) -> None:
        assert self.packet_handler, "Not connected"
        expected = int(enabled)
        result, error = self.packet_handler.write1ByteTxRx(
            actuator_id,
            ADDR_TORQUE_ENABLE,
            expected,
        )
        if result != 0 or error != 0:
            action = "enable" if enabled else "disable"
            raise RuntimeError(f"Failed to {action} torque for actuator {actuator_id}.")
        actual, result, error = self.packet_handler.read1ByteTxRx(
            actuator_id,
            ADDR_TORQUE_ENABLE,
        )
        if result != 0 or error != 0 or actual != expected:
            action = "enable" if enabled else "disable"
            raise RuntimeError(f"Torque {action} verification failed for actuator {actuator_id}.")

    def set_actuator_id(
        self,
        current_id: int,
        new_id: int,
        *,
        confirm: bool = True,
    ) -> bool:
        """Store a new EEPROM ID while retaining the active wire ID until power cycle."""
        self._validate_actuator_id(current_id, label="current")
        self._validate_actuator_id(new_id, label="new")
        if current_id not in self.actuator_configs:
            raise ValueError(f"Current Feetech actuator ID {current_id} is not configured.")
        if current_id == new_id:
            raise ValueError("Current and new Feetech actuator IDs must be different.")
        if new_id in self.actuator_configs:
            raise ValueError(f"Feetech actuator ID {new_id} is already configured.")
        return super().set_motor_id(current_id, new_id, confirm=confirm)

    @staticmethod
    def _validate_actuator_id(actuator_id: int, *, label: str) -> None:
        if not FEETECH_ACTUATOR_ID_MIN <= actuator_id <= FEETECH_ACTUATOR_ID_MAX:
            raise ValueError(
                f"Feetech {label} actuator ID must be between "
                f"{FEETECH_ACTUATOR_ID_MIN} and {FEETECH_ACTUATOR_ID_MAX}."
            )

    def angle_to_position(self, angle: float, actuator_id: int) -> int:
        angle = max(-math.pi, min(math.pi, angle))
        if self.actuator_configs[actuator_id].inverted:
            angle = -angle
        position = POS_MID + (angle / math.pi) * (POS_MAX - POS_MID)
        return int(position)

    def position_to_angle(self, position: int, actuator_id: int) -> float:
        angle = ((position - POS_MID) / (POS_MAX - POS_MID)) * math.pi
        if self.actuator_configs[actuator_id].inverted:
            angle = -angle
        return angle

    def write_position(  # ty: ignore[invalid-method-override]
        self,
        positions: dict[int, float],
        velocities: dict[int, float] | None = None,
        **kwargs,
    ) -> None:
        raw_positions = {
            actuator_id: self.angle_to_position(angle, actuator_id)
            for actuator_id, angle in positions.items()
        }
        requested_acceleration = kwargs.pop("acceleration", None)
        speed = kwargs.pop("speed", None)
        if kwargs:
            arguments = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Feetech position arguments: {arguments}.")
        if velocities is not None and not velocities.keys() <= positions.keys():
            raise ValueError(
                "Position trajectory velocities require matching position actuator IDs."
            )
        raw_speeds = {
            actuator_id: (
                self._trajectory_position_speed(
                    actuator_id,
                    positions[actuator_id],
                    velocities[actuator_id],
                )
                if velocities is not None and actuator_id in velocities
                else self._default_position_speed(actuator_id, speed)
            )
            for actuator_id in raw_positions
        }

        if requested_acceleration is not None:
            validate_feetech_acceleration(requested_acceleration)
            self._write_raw_positions(
                raw_positions,
                acceleration=requested_acceleration,
                speeds=raw_speeds,
            )
            return

        positions_by_acceleration: dict[int, dict[int, int]] = defaultdict(dict)
        for actuator_id, raw_position in raw_positions.items():
            acceleration = self.actuator_configs[actuator_id].max_acceleration
            positions_by_acceleration[acceleration][actuator_id] = raw_position
        for acceleration, grouped_positions in positions_by_acceleration.items():
            self._write_raw_positions(
                grouped_positions,
                acceleration=acceleration,
                speeds={actuator_id: raw_speeds[actuator_id] for actuator_id in grouped_positions},
            )

    @staticmethod
    def velocity_to_position_speed(velocity: float) -> int:
        """Convert a trajectory velocity in rad/s to an unsigned servo speed."""
        if not math.isfinite(velocity):
            raise ValueError("Feetech position velocity must be finite.")
        speed = round(abs(velocity) / SPEED_UNIT_RAD_S)
        return max(MIN_POSITION_SPEED, min(speed, DEFAULT_POSITION_SPEED))

    def _trajectory_position_speed(
        self,
        actuator_id: int,
        target_position: float,
        velocity: float,
    ) -> int:
        if not math.isfinite(velocity):
            raise ValueError("Feetech position velocity must be finite.")
        actuator_config = self.actuator_configs[actuator_id]
        measured_position = self._last_positions.get(actuator_id)
        if measured_position is None:
            return self.velocity_to_position_speed(actuator_config.max_position_velocity)

        tracking_error = abs(target_position - measured_position)
        requested_velocity = (
            abs(velocity) + actuator_config.position_tracking_error_gain * tracking_error
        )
        bounded_velocity = min(requested_velocity, actuator_config.max_position_velocity)
        return self.velocity_to_position_speed(bounded_velocity)

    def _default_position_speed(self, actuator_id: int, speed: int | None) -> int:
        if speed is not None:
            return speed
        configured_velocity = self.actuator_configs[actuator_id].max_position_velocity
        return self.velocity_to_position_speed(configured_velocity)

    def _write_raw_positions(
        self,
        raw_positions: dict[int, int],
        *,
        acceleration: int,
        speeds: dict[int, int],
    ) -> None:
        """Build and transmit one sync packet, checking the final bus result."""
        if not raw_positions:
            return
        assert self.packet_handler, "Not connected"
        validate_feetech_acceleration(acceleration)
        # The vendor SDK assigns one of two protocol handlers at runtime; their
        # SyncWritePosEx signatures differ by the HLS torque argument.
        packet_handler = cast(Any, self.packet_handler)
        sync_write = packet_handler.groupSyncWrite
        sync_write.clearParam()
        try:
            added = {}
            for actuator_id, raw_position in raw_positions.items():
                speed = speeds[actuator_id]
                if self.servo_type == "hls":
                    succeeded = packet_handler.SyncWritePosEx(
                        actuator_id,
                        raw_position,
                        speed,
                        acceleration,
                        HLS_FULL_TORQUE,
                    )
                else:
                    succeeded = packet_handler.SyncWritePosEx(
                        actuator_id,
                        raw_position,
                        speed,
                        acceleration,
                    )
                added[actuator_id] = bool(succeeded)
            self._raise_for_failed_writes("position parameter", added)

            result = sync_write.txPacket()
            if result != 0:
                actuator_ids = list(raw_positions)
                detail = packet_handler.getTxRxResult(result)
                issue = (
                    f"Feetech position sync write failed for actuator IDs {actuator_ids}: {detail}"
                )
                self._health_issues.update(dict.fromkeys(actuator_ids, issue))
                raise RuntimeError(issue)
        finally:
            sync_write.clearParam()

    def _raise_for_failed_writes(self, operation: str, results: dict[int, bool]) -> None:
        failed_ids = [actuator_id for actuator_id, succeeded in results.items() if not succeeded]
        if failed_ids:
            issue = f"Feetech {operation} write failed for actuator IDs {failed_ids}."
            self._health_issues.update(dict.fromkeys(failed_ids, issue))
            raise RuntimeError(issue)

    def read_position(self, actuator_id: int) -> float | None:  # ty: ignore[invalid-method-override]
        raw_position = super().read_position(actuator_id)
        if raw_position is None:
            return None
        position = self.position_to_angle(raw_position, actuator_id)
        self._last_positions[actuator_id] = position
        return position

    def read_all_positions(self) -> dict[int, float]:  # ty: ignore[invalid-method-override]
        raw_positions = super().read_all_positions()
        positions = {
            actuator_id: self.position_to_angle(position, actuator_id)
            for actuator_id, position in raw_positions.items()
        }
        self._last_positions.update(positions)
        return positions

    def probe(self, actuator_id: int) -> bool:
        """Return whether an ID responds, without logging expected misses."""
        assert self.packet_handler, "Not connected"
        _model, result, error = self.packet_handler.ping(actuator_id)
        return result == 0 and error == 0

    def ping(self, actuator_id: int) -> bool:
        try:
            if not self.probe(actuator_id):
                logger.error("No actuator found at ID %s. Check connections.", actuator_id)
                return False
            logger.info("Actuator found at ID %s", actuator_id)
            return True
        except Exception as exc:
            logger.error("Failed to read actuator at ID %s: %s", actuator_id, exc)
            return False

    def read_temperature(self, actuator_id: int) -> int:
        assert self.packet_handler, "Unable to read temperature"
        temperature, result, error = self.packet_handler.read1ByteTxRx(
            actuator_id,
            ADDR_TEMPERATURE,
        )
        if result != 0 or error != 0:
            raise RuntimeError(f"Problem reading temperature for actuator {actuator_id}")
        return temperature

    def read_all_temperatures(self) -> dict[int, float]:
        raw_temperatures = self._sync_read(
            ADDR_TEMPERATURE,
            TEMPERATURE_DATA_LENGTH,
            ((ADDR_TEMPERATURE, TEMPERATURE_DATA_LENGTH),),
            label="temperature",
        )
        return {actuator_id: float(values[0]) for actuator_id, values in raw_temperatures.items()}

    def write_velocity(self, velocities: dict[int, float]) -> None:
        assert self.packet_handler, "Not connected"
        for actuator_id, velocity in velocities.items():
            value = -velocity if self.actuator_configs[actuator_id].inverted else velocity
            magnitude = int(min(abs(value) / SPEED_UNIT_RAD_S, 32767))
            raw = (magnitude | 0x8000) if value < 0 else magnitude
            result, error = self.packet_handler.write2ByteTxRx(
                actuator_id,
                ADDR_GOAL_SPEED,
                raw,
            )
            if result != 0 or error != 0:
                issue = f"Feetech velocity write failed for actuator {actuator_id}."
                self._health_issues[actuator_id] = issue
                raise RuntimeError(issue)

    def read_velocity(self, actuator_id: int) -> float | None:
        assert self.packet_handler, "Not connected"
        raw, result, error = self.packet_handler.ReadSpeed(actuator_id)
        if result != 0 or error != 0:
            return None
        velocity = raw * SPEED_UNIT_RAD_S
        if self.actuator_configs[actuator_id].inverted:
            velocity = -velocity
        return velocity

    def read_all_velocities(self) -> dict[int, float]:
        raw_velocities = self._sync_read(
            ADDR_PRESENT_SPEED,
            SPEED_DATA_LENGTH,
            ((ADDR_PRESENT_SPEED, SPEED_DATA_LENGTH),),
            label="velocity",
        )
        return {
            actuator_id: self._velocity_from_raw(values[0], actuator_id)
            for actuator_id, values in raw_velocities.items()
        }

    def read_all_feedback(
        self,
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        """Read every actuator's present-state register block in one transaction."""
        raw_feedback = self._sync_read(
            ADDR_PRESENT_POSITION,
            PRESENT_FEEDBACK_DATA_LENGTH,
            (
                (ADDR_PRESENT_POSITION, POSITION_DATA_LENGTH),
                (ADDR_PRESENT_SPEED, SPEED_DATA_LENGTH),
                (ADDR_TEMPERATURE, TEMPERATURE_DATA_LENGTH),
            ),
            label="feedback",
        )
        positions = {
            actuator_id: self.position_to_angle(values[0], actuator_id)
            for actuator_id, values in raw_feedback.items()
        }
        self._last_positions.update(positions)
        velocities = {
            actuator_id: self._velocity_from_raw(values[1], actuator_id)
            for actuator_id, values in raw_feedback.items()
        }
        temperatures = {
            actuator_id: float(values[2]) for actuator_id, values in raw_feedback.items()
        }
        return positions, velocities, temperatures

    def health_issues(self) -> dict[int, str]:
        """Return the latest read or command issues keyed by actuator ID."""
        return dict(self._health_issues)

    def _velocity_from_raw(self, raw_velocity: int, actuator_id: int) -> float:
        assert self.packet_handler, "Not connected"
        signed_velocity = self.packet_handler.scs_tohost(raw_velocity, 15)
        velocity = signed_velocity * SPEED_UNIT_RAD_S
        return -velocity if self.actuator_configs[actuator_id].inverted else velocity

    def _sync_read(
        self,
        start_address: int,
        data_length: int,
        fields: tuple[tuple[int, int], ...],
        *,
        label: str,
    ) -> dict[int, tuple[int, ...]]:
        """Read one register range for all configured actuators."""
        assert self.packet_handler, "Not connected"
        group_sync_read = scs.GroupSyncRead(
            self.packet_handler,
            start_address,
            data_length,
        )
        actuator_ids = []
        health_issues: dict[int, str] = {}
        try:
            for actuator_id in self.actuator_ids:
                if group_sync_read.addParam(actuator_id):
                    actuator_ids.append(actuator_id)
                else:
                    health_issues[actuator_id] = (
                        f"Failed to add actuator to Feetech {label} sync read."
                    )

            result = group_sync_read.txRxPacket()
            if result != scs.COMM_SUCCESS:
                detail = self.packet_handler.getTxRxResult(result)
                issue = f"Feetech {label} sync read failed: {detail}"
                self._health_issues = dict.fromkeys(self.actuator_ids, issue)
                raise RuntimeError(f"Feetech {label} sync read failed: {detail}")

            values: dict[int, tuple[int, ...]] = {}
            for actuator_id in actuator_ids:
                available, error = group_sync_read.isAvailable(
                    actuator_id,
                    start_address,
                    data_length,
                )
                if error != 0:
                    detail = self.packet_handler.getRxPacketError(error)
                    health_issues[actuator_id] = f"Feetech {label} sync read reported: {detail}"
                    continue
                if not available:
                    health_issues[actuator_id] = f"Feetech {label} sync read returned no data."
                    continue
                values[actuator_id] = tuple(
                    group_sync_read.getData(actuator_id, address, length)
                    for address, length in fields
                )
            self._health_issues = health_issues
            for actuator_id, issue in health_issues.items():
                logger.warning("Actuator %s health issue: %s", actuator_id, issue)
            return values
        finally:
            group_sync_read.clearParam()
