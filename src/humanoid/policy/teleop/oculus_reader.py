"""Freshness-tracked adapter around the upstream Oculus ADB reader."""

import contextlib
import math
import socket
import subprocess
import threading
import time
from typing import Any

from oculus_reader import OculusReader as UpstreamOculusReader
from ppadb.connection import Connection

from humanoid.logger import get_logger
from humanoid.types.teleop import OculusInputSnapshot

logger = get_logger(__name__)


class OculusReader(UpstreamOculusReader):
    """Preserve capture times across the headset's buffered ADB log stream."""

    _stream_socket: socket.socket | None = None
    _reader_thread: threading.Thread | None = None

    def __init__(
        self,
        ip_address: str | None = None,
        port: int = 5555,
    ) -> None:
        self.last_captured_monotonic: float | None = None
        self._last_disengaged_monotonic: float | None = None
        super().__init__(ip_address=ip_address, port=port)

    def run(self) -> None:
        self.device.shell(
            f"am start -n {self.APK_name}/com.rail.oculus.teleop.MainActivity "
            "-a android.intent.action.MAIN -c android.intent.category.LAUNCHER"
        )
        self.running = True
        # Use the start of the clock query as a conservative bound: USB or network
        # delay can make a frame appear older, never younger than it really is.
        before_query = time.monotonic()
        device_epoch = float(self.device.shell("date +%s.%N"))
        if not math.isfinite(device_epoch):
            raise ValueError("Invalid clock sample from Oculus headset")
        self._device_to_monotonic = before_query - device_epoch
        self.device.shell(
            f'logcat -b main -T "{device_epoch:.6f}" -v threadtime -v epoch -s {self.tag}:V',
            self._start_stream,
        )

    def _start_stream(self, connection: Connection) -> None:
        self._stream_socket = connection.socket
        self._reader_thread = threading.Thread(
            target=self.read_logcat_by_line, args=(connection,), name="oculus-logcat", daemon=True
        )
        self._reader_thread.start()

    def stop(self) -> None:
        self.running = False
        if self._stream_socket is not None:
            # A silent headset leaves readline blocked until the socket wakes it.
            with contextlib.suppress(OSError):
                self._stream_socket.shutdown(socket.SHUT_RDWR)
        if (
            self._reader_thread is not None
            and self._reader_thread is not threading.current_thread()
        ):
            self._reader_thread.join(timeout=1)

    def get_network_device(self, client: Any, retry: int = 0) -> Any:
        """Connect to a wireless Quest with an actionable failure message."""
        del retry  # Compatibility with the upstream override point.
        address = f"{self.ip_address}:{self.port}"
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            try:
                subprocess.run(
                    ["adb", "start-server"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                raise RuntimeError(
                    "Could not start the local ADB server. Install adb and run "
                    "`adb start-server` before starting Oculus teleop."
                ) from exc
            client.remote_connect(self.ip_address, self.port)

        device = client.device(address)
        if device is None:
            raise RuntimeError(
                f"Could not connect to the Quest at {address}. Enable wireless ADB, "
                "confirm that the headset and this host share a network, and accept "
                "the headset's debugging authorization prompt."
            )
        return device

    def get_snapshot(self) -> OculusInputSnapshot:
        """Return controller data and its timestamp under the reader lock."""
        with self._lock:
            return OculusInputSnapshot(
                transforms=self.last_transforms,
                buttons=self.last_buttons,
                captured_monotonic=self.last_captured_monotonic,
                last_disengaged_monotonic=self._last_disengaged_monotonic,
            )

    def read_logcat_by_line(self, connection: Connection) -> None:
        """Read logcat records without treating a buffered record as a new pose."""
        marker = f" {self.tag}: "
        try:
            assert connection.socket is not None
            with connection.socket.makefile(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if not self.running:
                        break
                    try:
                        prefix, separator, payload = line.partition(marker)
                        if not separator:
                            continue
                        captured_at = float(prefix.split()[0]) + self._device_to_monotonic
                        transforms, buttons = self.process_data(payload.strip())
                    except (ValueError, IndexError):
                        continue
                    if transforms is None or buttons is None:
                        continue
                    with self._lock:
                        self.last_transforms = transforms
                        self.last_buttons = buttons
                        self.last_captured_monotonic = (
                            captured_at
                            if math.isfinite(captured_at) and captured_at <= time.monotonic()
                            else None
                        )
                        if (
                            "l" not in transforms
                            or "r" not in transforms
                            or not (buttons.get("LG") or buttons.get("RG"))
                        ):
                            self._last_disengaged_monotonic = self.last_captured_monotonic
        except OSError as error:
            if self.running:
                logger.warning("Oculus log stream disconnected: %s", error)
        finally:
            connection.close()
