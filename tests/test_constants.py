from pathlib import Path
from urllib.parse import urlsplit

from humanoid.constants import (
    DEFAULT_HUMANOID_ROBOT,
    DEFAULT_HUMANOID_RUNTIME,
    LCM_RECEIVE_BUFFER_BYTES,
    private_lcm_url,
)
from humanoid.types.process import Runtime
from humanoid.types.robot import RobotName


def test_default_humanoid_configuration_is_explicit():
    assert DEFAULT_HUMANOID_RUNTIME is Runtime.SIM
    assert DEFAULT_HUMANOID_ROBOT is RobotName.TRISKEL


def test_linux_deployment_matches_the_lcm_defaults():
    deploy = Path(__file__).resolve().parents[1] / "deploy"
    size = str(LCM_RECEIVE_BUFFER_BYTES)
    linux = (deploy / "90-humanoid-lcm.conf").read_text().replace(" ", "").splitlines()
    assert f"net.core.rmem_max={size}" in linux
    assert f"net.core.rmem_default={size}" in linux
    assert "LCM_DEFAULT_URL" not in (deploy / "humanoid.service").read_text()


def test_private_bus_reserves_room_for_all_channel_ports(monkeypatch):
    monkeypatch.setattr("humanoid.constants.secrets.randbelow", lambda n: n - 1)
    url = urlsplit(private_lcm_url())
    assert url.port is not None
    assert url.port + 499 <= 2**16 - 1
    assert url.hostname == "239.192.255.255"
    assert url.scheme == "mpudpm"
    assert url.query == f"ttl=0&recv_buf_size={LCM_RECEIVE_BUFFER_BYTES}"
