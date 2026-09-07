"""Freshness-tracked adapter around the upstream Oculus ADB reader."""

import subprocess
import time
from typing import Any

from oculus_reader import OculusReader as UpstreamOculusReader

from humanoid.types.teleop import OculusInputSnapshot


class OculusReader(UpstreamOculusReader):
    """Record when each controller frame arrives from the headset log stream."""

    def __init__(
        self,
        ip_address: str | None = None,
        port: int = 5555,
    ) -> None:
        self.last_received_monotonic: float | None = None
        super().__init__(ip_address=ip_address, port=port)

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
                received_monotonic=self.last_received_monotonic,
            )

    def read_logcat_by_line(self, connection: Any) -> None:
        """Read controller frames while retaining their local receipt time."""
        file_obj = connection.socket.makefile()
        try:
            while self.running:
                raw_line = file_obj.readline()
                if not raw_line:
                    break
                data = self.extract_data(raw_line.strip())
                if not data:
                    continue
                transforms, buttons = self.process_data(data)
                if transforms is None or buttons is None:
                    continue
                with self._lock:
                    self.last_transforms = transforms
                    self.last_buttons = buttons
                    self.last_received_monotonic = time.monotonic()
                if self.print_FPS:
                    self.fps_counter.getAndPrintFPS()
        finally:
            file_obj.close()
            connection.close()
