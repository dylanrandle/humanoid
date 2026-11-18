"""Thin environment wrapper for an injected remote control policy."""

from humanoid.constants import Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.nodes.base import Node
from humanoid.policy.base import Policy


class RemoteControlNode(Node):
    def __init__(self, policy: Policy, *, rate_hz: float) -> None:
        self.policy = policy
        self.rate_hz = rate_hz
        self.env = RealtimeEnvironment(
            ActionTopics(tool=Topic.AUTONOMY_TOOL_COMMAND, base=Topic.AUTONOMY_BASE_COMMAND),
        )

    def setup(self) -> None:
        self.env.reset()

    def step(self) -> None:
        observation = self.env.observe()
        action = self.policy(observation)
        self.env.step(action)

    def on_close(self) -> None:
        self.policy.reset()
        try:
            self.env.step(self.policy(self.env.observe()))
        finally:
            self.env.close()
