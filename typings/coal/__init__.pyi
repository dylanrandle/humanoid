from typing import Any

# Pinocchio's compiled collision library is installed beneath cmeel.prefix.
def __getattr__(name: str) -> Any: ...
