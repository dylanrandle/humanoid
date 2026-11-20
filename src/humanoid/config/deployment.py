"""Prepared model environments updated alongside the robot application."""

from pathlib import Path

MODEL_DIRECTORY = Path.home() / "humanoid-models"
MODEL_PYTHONS = (
    MODEL_DIRECTORY / ".venv/bin/python",
    MODEL_DIRECTORY / "graspgenx-venv/bin/python",
)
