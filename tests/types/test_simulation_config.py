"""Validation tests for native MuJoCo simulation configuration."""

from dataclasses import fields

import pytest

from humanoid.types.simulation import MujocoSimulationConfig

POSITIVE_SIMULATION_FIELDS = tuple(field.name for field in fields(MujocoSimulationConfig))


@pytest.mark.parametrize("field_name", POSITIVE_SIMULATION_FIELDS)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_simulation_values_must_be_positive_and_finite(field_name: str, value: float):
    with pytest.raises(ValueError, match="positive and finite"):
        MujocoSimulationConfig(**{field_name: value})
