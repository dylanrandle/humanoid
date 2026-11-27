"""Shared request parsing and response helpers for API routes."""

from http import HTTPStatus
from typing import Any

from flask import Response, jsonify, request
from werkzeug.exceptions import BadRequest

from humanoid.api.validation import parse_safety_context
from humanoid.config.api import JSON_CONTENT_TYPE, MAX_REQUEST_BYTES
from humanoid.types.api import ApiError, ApiKey, SafetyContext


def success(status: object) -> Response:
    return jsonify(ok=True, status=status)


def safety_context(payload: dict[str, Any]) -> SafetyContext:
    return parse_safety_context(
        payload.get(ApiKey.EXPECTED_RUNTIME),
        payload.get(ApiKey.EXPECTED_ROBOT),
        payload.get(ApiKey.REAL_HARDWARE_ACKNOWLEDGED),
    )


def json_payload() -> dict[str, Any]:
    if request.mimetype != JSON_CONTENT_TYPE:
        raise ApiError("Content-Type must be application/json.", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)

    content_length = request.content_length
    if content_length is not None and content_length > MAX_REQUEST_BYTES:
        raise ApiError("Request body is too large.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    if not content_length:
        raise ApiError("A JSON request body is required.", HTTPStatus.BAD_REQUEST)

    try:
        payload = request.get_json()
    except (BadRequest, UnicodeDecodeError) as exc:
        raise ApiError("Request body must be valid JSON.", HTTPStatus.BAD_REQUEST) from exc
    if not isinstance(payload, dict):
        raise ApiError("Request body must be a JSON object.", HTTPStatus.BAD_REQUEST)
    return payload
