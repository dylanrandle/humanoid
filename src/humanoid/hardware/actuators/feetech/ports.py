"""Select a servo adapter without opening ports or commanding hardware."""

from serial.tools import list_ports
from vassar_feetech_servo_sdk import find_servo_port
from vassar_feetech_servo_sdk.exceptions import PortNotFoundError


def detect_servo_port() -> str:
    """Prefer the WCH adapter used by this robot; reject ambiguous choices."""
    # USB identity observed on the robot's USB Single Serial adapter.
    # This identifies the adapter, not the attached servo model.
    adapter_usb_id = (0x1A86, 0x55D3)
    preferred = sorted(p.device for p in list_ports.comports() if (p.vid, p.pid) == adapter_usb_id)
    candidates = preferred or find_servo_port(return_all=True)
    if len(candidates) != 1:
        raise PortNotFoundError(
            f"Multiple possible servo adapters: {', '.join(candidates)}. "
            "Specify --port (or controller_config.port)."
        )
    return candidates[0]
