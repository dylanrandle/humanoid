"""Validation tests for native MuJoCo simulation configuration."""

from dataclasses import fields, replace

import pytest

from humanoid.types.simulation import MujocoSimulationConfig

POSITIVE_SIMULATION_FIELDS = tuple(
    field.name
    for field in fields(MujocoSimulationConfig)
    if field.name not in {"camera", "mesh_sliding_friction"}
)


@pytest.mark.parametrize("field_name", POSITIVE_SIMULATION_FIELDS)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_simulation_values_must_be_positive_and_finite(field_name: str, value: float):
    with pytest.raises(ValueError, match="positive and finite"):
        replace(MujocoSimulationConfig(), **{field_name: value})


@pytest.mark.parametrize(
    "friction",
    [{"fin": -1.0}, {"fin": float("nan")}, {"fin": float("inf")}, {"": 2.0}, {" ": 2.0}],
)
def test_mesh_friction_requires_named_finite_nonnegative_values(friction):
    with pytest.raises(ValueError, match="finite nonnegative"):
        MujocoSimulationConfig(mesh_sliding_friction=friction)


def test_mesh_friction_allows_frictionless_and_grip_tape_surfaces():
    config = MujocoSimulationConfig(mesh_sliding_friction={"frictionless": 0.0, "fin": 2.0})
    assert config.mesh_sliding_friction == {"frictionless": 0.0, "fin": 2.0}
