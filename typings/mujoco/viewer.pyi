"""Fallback typing surface for MuJoCo's passive native viewer."""

from typing import Any

class Handle:
    def is_running(self) -> bool: ...
    def sync(self) -> None: ...
    def close(self) -> None: ...

def launch_passive(
    model: Any,
    data: Any,
    *,
    key_callback: Any | None = ...,
    show_left_ui: bool = ...,
    show_right_ui: bool = ...,
) -> Handle: ...
