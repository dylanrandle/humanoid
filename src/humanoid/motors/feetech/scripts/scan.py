from humanoid.logger import get_logger
from humanoid.motors.feetech.configurator import FeetechMotorConfigurator

logger = get_logger(__name__)


def main():
    logger.info("Scanning for motors")
    results = FeetechMotorConfigurator.scan()
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
