import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from humanoid import start
from humanoid.api import server
from humanoid.constants import EnvironmentVariable


def test_spawned_oculus_worker_can_import_entrypoint_without_loading_console():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from humanoid.start import main\n"
            "from humanoid.nodes.policy.teleop.oculus import OculusTeleopNode\n"
            "assert 'humanoid.api.server' not in sys.modules\n"
            "assert 'humanoid.nodes.manager' not in sys.modules",
        ],
        check=True,
    )


def test_server_main_closes_service_and_web_server(monkeypatch):
    service = MagicMock()
    web_server = MagicMock(server_port=4321)
    web_server.serve_forever.side_effect = KeyboardInterrupt
    make_server = MagicMock(return_value=web_server)
    monkeypatch.setattr(server, "BackendApplication", MagicMock(return_value=service))
    monkeypatch.setattr(server, "make_server", make_server)
    monkeypatch.setattr(server.signal, "signal", MagicMock())
    monkeypatch.setattr(sys, "argv", ["start", "--no-open", "--port", "0"])

    start.main()

    assert make_server.call_args.args[:2] == ("127.0.0.1", 0)
    assert make_server.call_args.kwargs == {"threaded": True}
    web_server.serve_forever.assert_called_once_with()
    web_server.server_close.assert_called_once_with()
    service.close.assert_called_once_with()


def test_server_main_configures_wireless_oculus(monkeypatch):
    # Register an undo even when the variable was absent: main() writes it directly.
    monkeypatch.setenv(EnvironmentVariable.OCULUS_IP, "")
    service = MagicMock()
    web_server = MagicMock(server_port=4321)
    web_server.serve_forever.side_effect = KeyboardInterrupt
    monkeypatch.setattr(server, "BackendApplication", MagicMock(return_value=service))
    monkeypatch.setattr(server, "make_server", MagicMock(return_value=web_server))
    monkeypatch.setattr(server.signal, "signal", MagicMock())
    monkeypatch.setattr(
        sys,
        "argv",
        ["start", "--no-open", "--oculus-ip", "192.168.1.42"],
    )

    server.main()

    assert os.environ[EnvironmentVariable.OCULUS_IP] == "192.168.1.42"


def test_server_main_rejects_invalid_oculus_ip(monkeypatch):
    monkeypatch.setenv(EnvironmentVariable.OCULUS_IP, "")
    monkeypatch.setattr(sys, "argv", ["start", "--oculus-ip", "not-an-ip"])

    with pytest.raises(SystemExit):
        server.main()
