"""Validation tests for native MuJoCo simulation configuration."""

from dataclasses import fields

import pytest

from humanoid.constants import DEFAULT_MUJOCO_SCENE, MUJOCO_SCENE_ENVIRONMENT_VARIABLE
from humanoid.types.simulation import (
    FloorAndCubeSceneConfig,
    MujocoScene,
    MujocoSimulationConfig,
)

POSITIVE_SIMULATION_FIELDS = tuple(field.name for field in fields(MujocoSimulationConfig))


@pytest.mark.parametrize("field_name", POSITIVE_SIMULATION_FIELDS)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_simulation_values_must_be_positive_and_finite(field_name: str, value: float):
    with pytest.raises(ValueError, match="positive and finite"):
        MujocoSimulationConfig(**{field_name: value})


@pytest.mark.parametrize("field_name", ["floor_half_extent", "cube_edge_length", "cube_mass"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_scene_dimensions_and_mass_must_be_positive_and_finite(
    field_name: str,
    value: float,
):
    with pytest.raises(ValueError, match="positive and finite"):
        FloorAndCubeSceneConfig(**{field_name: value})


@pytest.mark.parametrize("field_name", ["cube_x", "cube_y"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_cube_position_must_be_finite(field_name: str, value: float):
    with pytest.raises(ValueError, match="position must be finite"):
        FloorAndCubeSceneConfig(**{field_name: value})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DEFAULT_MUJOCO_SCENE),
        ("", DEFAULT_MUJOCO_SCENE),
        ("empty", MujocoScene.EMPTY),
        ("floor-and-cube", MujocoScene.FLOOR_AND_CUBE),
        ("FLOOR-AND-CUBE", MujocoScene.FLOOR_AND_CUBE),
    ],
)
def test_scene_environment_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(MUJOCO_SCENE_ENVIRONMENT_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(MUJOCO_SCENE_ENVIRONMENT_VARIABLE, value)

    assert MujocoScene.from_environment() is expected


def test_invalid_scene_environment_is_rejected(monkeypatch):
    monkeypatch.setenv(MUJOCO_SCENE_ENVIRONMENT_VARIABLE, "warehouse")

    with pytest.raises(ValueError, match="warehouse"):
        MujocoScene.from_environment()
