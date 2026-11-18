"""Project defaults for the remote control policy."""

from humanoid.types.remote_control import RemoteControlPolicyConfig


def get_remote_control_policy_config() -> RemoteControlPolicyConfig:
    return RemoteControlPolicyConfig()
