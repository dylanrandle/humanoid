"""Managed nodes for the agent policy and HTTP control server."""

from humanoid.agent.server import AgentControlServer
from humanoid.agent.service import AgentControlService
from humanoid.config import ROBOT_CONFIG
from humanoid.config.agent import get_agent_control_server_config, get_agent_policy_config
from humanoid.constants import DEFAULT_LCM_URL, Topic
from humanoid.environment.realtime import ActionTopics, RealtimeEnvironment
from humanoid.nodes.base import Node
from humanoid.orchestrator.client import OrchestratorClient
from humanoid.policy.agent import AgentPolicy
from humanoid.policy.base import Policy
from humanoid.robots.base import Robot
from humanoid.types.agent import AgentControlServerConfig, AgentPolicyConfig
from humanoid.types.observation import ObservationSubscriptions
from humanoid.types.robot import RobotConfig


class AgentPolicyNode(Node):
    def __init__(
        self,
        policy: Policy,
        *,
        rate_hz: float,
        lcm_url: str = DEFAULT_LCM_URL,
        subscriptions: ObservationSubscriptions,
    ) -> None:
        self.policy = policy
        self.rate_hz = rate_hz
        self._ready = False
        self.env = RealtimeEnvironment(
            ActionTopics(tool=Topic.AUTONOMY_TOOL_COMMAND, base=Topic.AUTONOMY_BASE_COMMAND),
            lcm_url=lcm_url,
            subscriptions=subscriptions,
        )

    def setup(self) -> None:
        self.env.reset(state_timeout_ms=10000)
        self._ready = True

    def step(self) -> None:
        observation = self.env.observe()
        action = self.policy(observation)
        self.env.step(action)

    def on_close(self) -> None:
        self.policy.reset()
        try:
            if self._ready:
                self.env.step(self.policy(self.env.observe()))
        finally:
            self.env.close()


class AgentControlServerNode(AgentPolicyNode):
    """Serve HTTP while the node runs the policy on its control loop."""

    def __init__(
        self,
        server_config: AgentControlServerConfig | None = None,
        robot_config: RobotConfig = ROBOT_CONFIG,
        config: AgentPolicyConfig | None = None,
        lcm_url: str = DEFAULT_LCM_URL,
    ) -> None:
        config = config or get_agent_policy_config(robot_config)
        policy = AgentPolicy(Robot(robot_config), config=config)
        super().__init__(
            policy,
            rate_hz=1.0 / config.dt,
            lcm_url=lcm_url,
            subscriptions=config.subscriptions,
        )
        self.service = AgentControlService(
            policy, server_config or get_agent_control_server_config()
        )
        self.server = AgentControlServer(
            self.service, OrchestratorClient(self.env.publisher).release_autonomy
        )

    def setup(self) -> None:
        super().setup()
        self.service.observe(self.env.observe())
        self.server.start()

    def step(self) -> None:
        observation = self.env.observe()
        self.service.observe(observation)
        self.env.step(self.policy(observation))

    def on_close(self) -> None:
        try:
            self.server.close()
        finally:
            super().on_close()
