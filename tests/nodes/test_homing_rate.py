from humanoid.config.homing import HOMING_POLICY_CONFIG

EXPECTED_HOMING_RATE_HZ = 30.0


def test_default_homing_rate_is_thirty_hz():
    assert HOMING_POLICY_CONFIG.rate_hz == EXPECTED_HOMING_RATE_HZ
