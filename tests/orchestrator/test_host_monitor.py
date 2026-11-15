from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from humanoid.orchestrator.monitor import host


@pytest.fixture
def sampling(monkeypatch):
    clock = Mock(return_value=0.0)
    cpu = Mock()
    cpu.return_value._asdict.return_value = {"user": 10.0, "idle": 90.0}
    memory = Mock(return_value=SimpleNamespace(total=8192, available=6144, percent=25.0))
    monkeypatch.setattr(host.psutil, "cpu_times", cpu)
    monkeypatch.setattr(host.psutil, "virtual_memory", memory)
    return host.HostResourceMonitor(clock=clock), clock, cpu, memory


def test_cpu_warms_up_and_memory_uses_available_pages(sampling):
    monitor, clock, cpu, memory = sampling
    first = monitor.snapshot()
    assert first is not None
    assert first.cpu_percent is None
    assert first.memory_used_bytes == memory.return_value.total - memory.return_value.available
    assert first.memory_total_bytes == memory.return_value.total
    assert first.memory_percent == memory.return_value.percent
    clock.return_value = 0.5
    assert monitor.snapshot() is first
    memory.assert_called_once()
    cpu.assert_called_once()

    clock.return_value = 1.0
    cpu.return_value._asdict.return_value = {"user": 30.0, "idle": 170.0}
    second = monitor.snapshot()
    assert second is not None
    assert second.cpu_percent == pytest.approx(20.0)


def test_linux_guest_is_not_double_counted_and_iowait_is_idle(sampling):
    monitor, clock, cpu, _ = sampling
    cpu.return_value._asdict.return_value = {
        "user": 100.0,
        "idle": 100.0,
        "iowait": 10.0,
        "guest": 30.0,
        "guest_nice": 0.0,
    }
    monitor.snapshot()
    clock.return_value = 1.0
    cpu.return_value._asdict.return_value = {
        "user": 120.0,
        "idle": 160.0,
        "iowait": 30.0,
        "guest": 40.0,
        "guest_nice": 0.0,
    }
    status = monitor.snapshot()
    assert status is not None
    assert status.cpu_percent == pytest.approx(20.0)


def test_read_failure_hides_old_metrics_and_recovers(sampling):
    monitor, clock, cpu, memory = sampling
    assert monitor.snapshot() is not None
    clock.return_value = 1.0
    memory.side_effect = OSError("host telemetry unavailable")
    assert monitor.snapshot() is None
    clock.return_value = 2.0
    memory.side_effect = None
    cpu.return_value._asdict.return_value = {"user": 40.0, "idle": 160.0}
    recovered = monitor.snapshot()
    assert recovered is not None
    assert recovered.cpu_percent is None


def test_counter_reset_never_reports_negative_or_overfull_cpu(sampling):
    monitor, clock, cpu, _ = sampling
    monitor.snapshot()
    clock.return_value = 1.0
    cpu.return_value._asdict.return_value = {"user": 9.0, "idle": 89.0}
    status = monitor.snapshot()
    assert status is not None
    assert status.cpu_percent is None
