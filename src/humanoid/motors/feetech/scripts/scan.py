from humanoid.logging import get_logger
from humanoid.motors.feetech.controller import FeetechConfigurator

logger = get_logger(__name__)


def main():
    logger.info("Scanning for motors")
    results = FeetechConfigurator.scan()
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
