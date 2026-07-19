"""Imported node groups exposed to the process manager."""

from humanoid.nodes.orchestrator import OrchestratorNode
from humanoid.nodes.policy.homing import HomingNode
from humanoid.nodes.policy.teleop.keyboard import KeyboardTeleopNode
from humanoid.nodes.policy.teleop.oculus import OculusTeleopNode
from humanoid.nodes.robot.controller import RobotControllerNode
from humanoid.nodes.robot.driver import RobotDriverNode
from humanoid.nodes.robot.logger import RobotLoggerNode
from humanoid.nodes.robot.simulation import MujocoSimulationNode
from humanoid.nodes.robot.visualizer import RobotVisualizerNode
from humanoid.types.node import NodeGroup
from humanoid.types.process import ProcessName, Runtime

NODE_GROUPS = {
    group.name: group
    for group in (
        NodeGroup(
            name=ProcessName.STACK,
            display_name="Main stack",
            nodes=(
                RobotControllerNode,
                RobotVisualizerNode,
                OrchestratorNode,
                RobotLoggerNode,
            ),
            deferred_nodes=(HomingNode,),
            runtime_nodes={
                Runtime.SIM: (MujocoSimulationNode,),
                Runtime.REAL: (RobotDriverNode,),
            },
        ),
        NodeGroup(
            name=ProcessName.REPLAY,
            display_name="Replay runtime",
            nodes=(RobotVisualizerNode,),
            runtime_nodes={
                Runtime.SIM: (MujocoSimulationNode,),
                Runtime.REAL: (RobotDriverNode,),
            },
        ),
        NodeGroup(
            name=ProcessName.KEYBOARD,
            display_name="Keyboard teleop",
            nodes=(KeyboardTeleopNode,),
        ),
        NodeGroup(
            name=ProcessName.OCULUS,
            display_name="Oculus teleop",
            nodes=(OculusTeleopNode,),
        ),
    )
}
PROCESS_ORDER = tuple(NODE_GROUPS)
PROCESS_STOP_ORDER = (
    ProcessName.KEYBOARD,
    ProcessName.OCULUS,
    ProcessName.REPLAY,
    ProcessName.STACK,
)


def process_display_name(name: ProcessName) -> str:
    return NODE_GROUPS[name].display_name
