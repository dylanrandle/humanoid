"""Tests for the freshness-tracked Oculus reader adapter."""

import io
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from humanoid.policy.teleop.oculus_reader import OculusReader


def _reader_without_device() -> OculusReader:
    reader = object.__new__(OculusReader)
    reader.ip_address = "192.168.1.42"
    reader.port = 5555
    reader.running = True
    reader.print_FPS = False
    reader.last_transforms = {}
    reader.last_buttons = {}
    reader.last_received_monotonic = None
    reader._lock = threading.Lock()
    return reader


def test_wireless_device_connects_by_address():
    reader = _reader_without_device()
    client = MagicMock()
    device = object()
    client.device.return_value = device

    result = reader.get_network_device(client)

    assert result is device
    client.remote_connect.assert_called_once_with("192.168.1.42", 5555)
    client.device.assert_called_once_with("192.168.1.42:5555")


def test_wireless_device_reports_unreachable_address():
    reader = _reader_without_device()
    client = MagicMock()
    client.device.return_value = None

    with pytest.raises(RuntimeError, match=r"192\.168\.1\.42:5555"):
        reader.get_network_device(client)


def test_logcat_frame_records_local_receipt_time(monkeypatch):
    reader = _reader_without_device()
    connection = MagicMock()
    connection.socket.makefile.return_value = io.StringIO("controller frame\n")
    transforms = {"r": np.eye(4)}
    buttons = {"RG": False}
    monkeypatch.setattr(reader, "extract_data", MagicMock(return_value="payload"))
    monkeypatch.setattr(
        reader,
        "process_data",
        MagicMock(return_value=(transforms, buttons)),
    )

    reader.read_logcat_by_line(connection)
    snapshot = reader.get_snapshot()

    assert snapshot.transforms is transforms
    assert snapshot.buttons is buttons
    assert snapshot.received_monotonic is not None
    connection.close.assert_called_once_with()
