"""Wire conversion for recording requests."""

from humanoid.types.lcm import logging_request_t
from humanoid.types.logging import LoggingRequest


def logging_request_to_lcm(value: LoggingRequest) -> logging_request_t:
    message = logging_request_t()
    message.timestamp = round(value.timestamp * 1e9)
    message.start = value.start
    message.directory = value.directory
    return message


def logging_request_from_lcm(message: logging_request_t) -> LoggingRequest:
    return LoggingRequest(message.timestamp / 1e9, bool(message.start), message.directory)
