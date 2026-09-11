import time
from collections.abc import Callable


def _next_deadline(previous_deadline: float, period: float, now: float) -> float:
    """Advance one period, skipping deadlines that have already elapsed."""
    deadline = previous_deadline + period
    if deadline <= now:
        missed_periods = int((now - deadline) // period) + 1
        deadline += missed_periods * period
    return deadline


def loop_at_rate(
    func: Callable[[], None],
    rate_hz: float,
    duration: float | None = None,
    stop_condition: Callable[[], bool] | None = None,
) -> None:
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")

    period = 1.0 / rate_hz
    start_time = time.perf_counter()
    next_call_time = start_time

    while True:
        if duration is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= duration:
                break

        if stop_condition is not None and stop_condition():
            break

        # Execute the function
        func()

        current_time = time.perf_counter()
        next_call_time = _next_deadline(next_call_time, period, current_time)
        sleep_time = next_call_time - current_time

        time.sleep(sleep_time)
