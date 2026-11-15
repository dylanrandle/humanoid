"""Host-wide resource telemetry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HostResourceStatus:
    """CPU across all cores and memory excluding available/reclaimable pages."""

    cpu_percent: float | None
    memory_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
