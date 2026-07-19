import pytest

from humanoid.constants import DEFAULT_HUMANOID_RUNTIME, RUNTIME_ENVIRONMENT_VARIABLE
from humanoid.types.process import Runtime


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DEFAULT_HUMANOID_RUNTIME),
        ("", DEFAULT_HUMANOID_RUNTIME),
        ("sim", Runtime.SIM),
        ("real", Runtime.REAL),
        ("REAL", Runtime.REAL),
    ],
)
def test_runtime_environment_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(RUNTIME_ENVIRONMENT_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(RUNTIME_ENVIRONMENT_VARIABLE, value)

    assert Runtime.from_environment() is expected


@pytest.mark.parametrize("value", ["simulation", "typo"])
def test_invalid_runtime_environment_is_rejected(monkeypatch, value):
    monkeypatch.setenv(RUNTIME_ENVIRONMENT_VARIABLE, value)

    with pytest.raises(ValueError, match=value):
        Runtime.from_environment()
