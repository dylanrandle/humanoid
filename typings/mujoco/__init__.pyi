"""Fallback typing surface for MuJoCo's dynamically exposed Python API."""

from typing import Any

def __getattr__(name: str) -> Any: ...
