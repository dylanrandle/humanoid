"""Logging configuration for the humanoid project."""

import logging
import sys
from logging.handlers import QueueHandler
from multiprocessing.queues import Queue

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(processName)s %(filename)s:%(lineno)d: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
) -> None:
    formatter = create_log_formatter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)


def setup_queue_logging(log_queue: Queue[logging.LogRecord]) -> None:
    """Send all child-process records to the parent logging listener."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(QueueHandler(log_queue))


def create_log_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
