from humanoid.nodes.policy.homing import DEFAULT_RATE_HZ

EXPECTED_HOMING_RATE_HZ = 30.0


def test_default_homing_rate_is_thirty_hz():
    assert DEFAULT_RATE_HZ == EXPECTED_HOMING_RATE_HZ
