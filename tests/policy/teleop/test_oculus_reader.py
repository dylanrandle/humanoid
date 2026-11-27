"""Tests for the freshness-tracked Oculus reader adapter."""

import io
import socket
import threading
import time
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
    reader.tag = "wE9ryARX"
    reader.last_transforms = {}
    reader.last_buttons = {}
    reader.last_captured_monotonic = None
    reader._last_disengaged_monotonic = None
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


def test_buffered_logcat_frame_keeps_its_original_capture_time(monkeypatch: pytest.MonkeyPatch):
    # Reproduce the clock-offset roundoff seen on CI regardless of host uptime.
    monkeypatch.setattr(time, "monotonic", lambda: 144.287430623)
    reader = _reader_without_device()
    expected_capture = time.monotonic() - 25
    reader._device_to_monotonic = expected_capture - 1000
    connection = MagicMock()
    connection.socket.makefile.return_value = io.StringIO(
        "--------- beginning of main\n"
        "999.500 100 101 I wE9ryARX: r:bad&R\n"
        "1000.000 100 101 I wE9ryARX: r:1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1&R,RG\n"
    )

    reader.read_logcat_by_line(connection)
    snapshot = reader.get_snapshot()

    np.testing.assert_array_equal(snapshot.transforms["r"], np.eye(4))
    assert snapshot.buttons["RG"]
    assert snapshot.captured_monotonic == pytest.approx(expected_capture, rel=0, abs=1e-9)
    connection.close.assert_called_once_with()


def test_release_is_visible_even_if_reader_has_already_received_the_next_grip():
    reader = _reader_without_device()
    reader._device_to_monotonic = time.monotonic() - 1003
    pose = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    packet = f"100 101 I wE9ryARX: l:{pose}|r:{pose}&L,R"
    connection = MagicMock()
    connection.socket.makefile.return_value = io.StringIO(
        f"1000.000 {packet},RG\n1001.000 {packet}\n1002.000 {packet},RG\n"
    )
    reader.read_logcat_by_line(connection)
    snapshot = reader.get_snapshot()
    assert snapshot.buttons["RG"]
    assert snapshot.last_disengaged_monotonic == reader._device_to_monotonic + 1001
    assert snapshot.captured_monotonic == reader._device_to_monotonic + 1002


def test_filtered_stream_stops_while_headset_is_silent():
    reader = _reader_without_device()
    reader.device = MagicMock()
    reader.APK_name = "com.rail.oculus.teleop"
    reader.tag = "wE9ryARX"
    host, headset = socket.socketpair()
    with host, headset:
        connection = MagicMock()
        connection.socket = host
        reader.device.shell.side_effect = lambda cmd, handler=None: (
            handler(connection) if handler is not None else "1000.000"
        )
        reader.run()
        reader.device.shell.assert_called_with(
            'logcat -b main -T "1000.000000" -v threadtime -v epoch -s wE9ryARX:V',
            reader._start_stream,
        )
        thread = reader._reader_thread
        assert thread is not None and thread.is_alive()
        reader.stop()
        assert not thread.is_alive()
        connection.close.assert_called_once_with()
