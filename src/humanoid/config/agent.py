"""Project defaults for the agent policy and local control API."""

import os

from humanoid.types.agent import AgentControlEndpoint, AgentControlServerConfig, AgentPolicyConfig

AGENT_HOST = "127.0.0.1"
DEFAULT_AGENT_PORT = 8767
MAX_TCP_PORT = 65535
DEFAULT_AGENT_ENDPOINT = AgentControlEndpoint(f"http://{AGENT_HOST}:{DEFAULT_AGENT_PORT}")


def get_agent_policy_config() -> AgentPolicyConfig:
    return AgentPolicyConfig()


def get_agent_control_server_config() -> AgentControlServerConfig:
    port = int(os.getenv("HUMANOID_CONTROL_PORT", DEFAULT_AGENT_PORT))
    if not 1 <= port <= MAX_TCP_PORT:
        raise ValueError(f"HUMANOID_CONTROL_PORT must be between 1 and {MAX_TCP_PORT}.")
    return AgentControlServerConfig(port=port)
