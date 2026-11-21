"""Fallback typing surface for MuJoCo's passive native viewer."""

from contextlib import AbstractContextManager

from mujoco import MjData, MjModel

class Handle(AbstractContextManager["Handle"]):
    def is_running(self) -> bool: ...
    def sync(self) -> None: ...

def launch_passive(model: MjModel, data: MjData) -> Handle: ...
