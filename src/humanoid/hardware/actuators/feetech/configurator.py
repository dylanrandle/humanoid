from humanoid.hardware.actuators.feetech.config import (
    FEETECH_ACTUATOR_ID_MAX,
    FEETECH_ACTUATOR_ID_MIN,
    FeetechActuatorControllerConfig,
)
from humanoid.hardware.actuators.feetech.driver import FeetechActuatorDriver
from humanoid.logger import get_logger

logger = get_logger(__name__)

ADDR_P_GAIN = 21
ADDR_D_GAIN = 22
ADDR_I_GAIN = 23
ADDR_LOCK = 48


class FeetechActuatorConfigurator:
    @classmethod
    def scan(
        cls,
        controller_config: FeetechActuatorControllerConfig | None = None,
    ) -> list[int]:
        found_ids = []
        possible_ids = list(range(FEETECH_ACTUATOR_ID_MIN, FEETECH_ACTUATOR_ID_MAX + 1))
        with FeetechActuatorDriver.for_actuator_ids(
            [],
            controller_config=controller_config,
        ) as driver:
            for checked, actuator_id in enumerate(possible_ids, start=1):
                if driver.probe(actuator_id):
                    found_ids.append(actuator_id)
                logger.info("Checked: %s/%s", checked, len(possible_ids))
        return found_ids

    @classmethod
    def set_id(
        cls,
        current_id: int,
        new_id: int,
        controller_config: FeetechActuatorControllerConfig | None = None,
    ) -> bool:
        for label, actuator_id in (("current", current_id), ("new", new_id)):
            if not (FEETECH_ACTUATOR_ID_MIN <= actuator_id <= FEETECH_ACTUATOR_ID_MAX):
                logger.error(
                    f"Invalid {label} ID: {actuator_id}. Must be between "
                    f"{FEETECH_ACTUATOR_ID_MIN} and {FEETECH_ACTUATOR_ID_MAX}."
                )
                return False

        if current_id == new_id:
            logger.warning(f"Current ID and new ID are the same ({current_id}). No change needed.")
            return True

        logger.info(f"Attempting to change actuator ID from {current_id} to {new_id}...")

        with FeetechActuatorDriver.for_actuator_ids(
            [current_id],
            controller_config=controller_config,
        ) as driver:
            if not driver.ping(current_id):
                logger.error(f"Unable to find actuator {current_id}")
                return False
            if driver.probe(new_id):
                logger.error(
                    f"Actuator ID {new_id} already responds on the selected controller bus"
                )
                return False

            success = driver.set_actuator_id(current_id, new_id)

            if success:
                logger.info(f"✓ Successfully changed actuator ID from {current_id} to {new_id}")
            else:
                logger.error(f"Failed to change actuator ID from {current_id} to {new_id}")

            return success

    @classmethod
    def set_zero(
        cls,
        actuator_id: int,
        controller_config: FeetechActuatorControllerConfig | None = None,
    ) -> None:
        logger.info(f"Setting zero (middle) position for {actuator_id}")
        with FeetechActuatorDriver.for_actuator_ids(
            [actuator_id],
            controller_config=controller_config,
            configure_on_connect=True,
        ) as driver:
            driver.set_middle_position([actuator_id])

    @classmethod
    def read_gains(
        cls,
        actuator_id: int,
        controller_config: FeetechActuatorControllerConfig | None = None,
    ) -> dict[str, int]:
        gains = {}
        with FeetechActuatorDriver.for_actuator_ids(
            [actuator_id],
            controller_config=controller_config,
        ) as driver:
            for addr, name in zip(
                [ADDR_P_GAIN, ADDR_I_GAIN, ADDR_D_GAIN], ["P", "I", "D"], strict=True
            ):
                assert driver.packet_handler
                curr, res, err = driver.packet_handler.read1ByteTxRx(actuator_id, addr)
                if res != 0 or err != 0:
                    raise RuntimeError(f"Problem reading {name} gain for {actuator_id}")
                logger.info(f"{name} gain for {actuator_id}: {curr}")
                gains[name.lower()] = curr
        return gains
