import time
from collections.abc import Callable


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

    iteration = 0
    while True:
        if duration is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= duration:
                break

        if stop_condition is not None and stop_condition():
            break

        # Execute the function
        func()

        # Calculate next call time
        iteration += 1
        next_call_time = start_time + (iteration * period)

        current_time = time.perf_counter()
        sleep_time = next_call_time - current_time

        if sleep_time > 0:
            time.sleep(sleep_time)
