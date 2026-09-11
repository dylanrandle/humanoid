import pytest

from humanoid.config import (
    get_keyboard_teleop_policy_config,
    get_oculus_teleop_policy_config,
)

EXPECTED_TELEOP_RATE_HZ = 30.0


@pytest.mark.parametrize(
    "config_factory",
    [get_keyboard_teleop_policy_config, get_oculus_teleop_policy_config],
)
def test_runtime_teleop_config_targets_thirty_hz(config_factory):
    config = config_factory()

    assert 1.0 / config.dt == pytest.approx(EXPECTED_TELEOP_RATE_HZ)
