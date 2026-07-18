import sys
from unittest.mock import MagicMock

from humanoid import start
from humanoid.ui import server


def test_start_command_launches_operator_console():
    assert start.main is server.main


def test_server_main_closes_service_and_web_server(monkeypatch):
    service = MagicMock()
    web_server = MagicMock(server_port=4321)
    web_server.serve_forever.side_effect = KeyboardInterrupt
    make_server = MagicMock(return_value=web_server)
    monkeypatch.setattr(server, "OrchestratorService", MagicMock(return_value=service))
    monkeypatch.setattr(server, "make_server", make_server)
    monkeypatch.setattr(server.signal, "signal", MagicMock())
    monkeypatch.setattr(sys, "argv", ["start", "--no-open", "--port", "0"])

    server.main()

    assert make_server.call_args.args[:2] == ("127.0.0.1", 0)
    assert make_server.call_args.kwargs == {"threaded": True}
    web_server.serve_forever.assert_called_once_with()
    web_server.server_close.assert_called_once_with()
    service.close.assert_called_once_with()
