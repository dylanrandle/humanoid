import pytest

from humanoid.config.robot.triskel import OPERATIONAL_SPACE_CONFIG

EXPECTED_CONTROLLER_RATE_HZ = 10.0


def test_triskel_controller_targets_ten_hz():
    assert 1 / OPERATIONAL_SPACE_CONFIG.dt == pytest.approx(EXPECTED_CONTROLLER_RATE_HZ)
