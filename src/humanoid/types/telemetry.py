"""Channel telemetry exposed by the backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LCMChannelStatus:
    """Observed bus traffic, without assuming a target rate for event channels."""

    channel: str
    publish_rate_hz: float
    age_seconds: float
    active: bool
