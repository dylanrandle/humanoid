"""Errors produced by the operator-console HTTP boundary."""

from http import HTTPStatus


class ApiError(Exception):
    """An expected API failure that is safe to return to the caller."""

    def __init__(self, message: str, status: HTTPStatus):
        super().__init__(message)
        self.status = status
