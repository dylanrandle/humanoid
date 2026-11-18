"""Operator defaults for the standalone control API."""

from humanoid.types.remote_api import RemoteControlServerConfig


def get_remote_control_server_config(token: str) -> RemoteControlServerConfig:
    return RemoteControlServerConfig(host="127.0.0.1", port=8767, token=token)
