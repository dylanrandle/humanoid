"""One background console operation. The control service owns all robot motion."""

import signal
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from multiprocessing.connection import Connection

from openai_codex import Codex, Thread

from humanoid.agent.client import AgentControlClient
from humanoid.agent.harness.model import connect, decide, open_codex, start_thread
from humanoid.agent.harness.robot import RobotTools, image_url
from humanoid.config.agent import (
    MAX_AGENT_COMMANDS,
    MAX_AGENT_PERCEPTION_CALLS,
    get_agent_endpoint,
    get_agent_policy_config,
)
from humanoid.config.robot import ROBOT_CONFIGS
from humanoid.robots.base import Robot
from humanoid.types.agent import (
    AgentPanelStatus,
    AgentPhase,
    AgentTask,
    ControlOperation,
    DecisionContext,
    MotionArguments,
    ObserveAgain,
    ToolError,
)
from humanoid.types.orchestrator import Mode
from humanoid.types.tools import CompleteDepth


def run_task(
    codex: Codex, client: AgentControlClient, task: AgentTask, update: Callable[[str], None]
) -> str:
    if client.session_id != task.session_id:
        raise RuntimeError("Agent API restarted. Submit a new task.")
    with closing(RobotTools(client, Robot(ROBOT_CONFIGS[task.robot]))) as tools:
        thread = start_thread(codex, task.model)
        return _run_steps(thread, tools, client, task, update)


def _run_steps(
    thread: Thread,
    tools: RobotTools,
    client: AgentControlClient,
    task: AgentTask,
    update: Callable[[str], None],
) -> str:
    count = perception = 0
    status, images = tools.observe(task.mode)
    feedback = None
    while True:
        allow_motion = task.mode is Mode.AUTONOMY and count < MAX_AGENT_COMMANDS
        context = DecisionContext(
            instruction=task.instruction,
            observation=status,
            last_result=feedback,
            commands_remaining=MAX_AGENT_COMMANDS - count if allow_motion else 0,
            perception_remaining=MAX_AGENT_PERCEPTION_CALLS - perception,
            limits=get_agent_policy_config(),
        )
        decision = decide(
            thread,
            context,
            images,
            allow_motion=allow_motion,
        )
        command = decision.command
        if command is None:
            return decision.message
        name = command.operation
        args = command.arguments
        if not isinstance(args, MotionArguments):
            if perception >= MAX_AGENT_PERCEPTION_CALLS:
                raise RuntimeError("Perception call budget exhausted.")
            perception += 1
            if client.observation().mode is not task.mode:
                raise RuntimeError("Robot mode changed during the task.")
            update(f"{decision.message}\nRunning {name}.")
            if isinstance(args, ObserveAgain):
                status, images = tools.observe(task.mode)
                feedback = None
            else:
                try:
                    if isinstance(args, CompleteDepth):
                        feedback = tools.depth.complete_depth(args)
                    else:
                        feedback, preview = tools.segmentation.segment(args)
                        images["Perception result for " + feedback.observation_id] = image_url(
                            preview
                        )
                except (ValueError, OSError, RuntimeError, ImportError) as error:
                    feedback = ToolError(operation=name, error=str(error))
            continue
        if not allow_motion or client.observation().mode is not Mode.AUTONOMY:
            raise RuntimeError("Motion requires an active Autonomy task.")
        operation = ControlOperation(name)
        update(f"{decision.message}\nExecuting {operation}.")
        feedback = client.execute(operation, args.model_dump(mode="json"))
        update(f"{operation} finished (reached={feedback['reached']}). Observing again.")
        count += 1
        status, images = tools.observe(task.mode)


def _stop(signum, frame) -> None:
    # Unwind execute() and close its in-flight command before exiting.
    raise SystemExit


def work(channel: Connection, task: AgentTask | None, initial_status: AgentPanelStatus) -> None:
    signal.signal(signal.SIGTERM, _stop)

    def show_device_code(url: str, code: str) -> None:
        channel.send(
            replace(
                initial_status,
                login_url=url,
                login_code=code,
                message="Enter the device code to finish signing in.",
            )
        )

    def show_progress(message: str) -> None:
        channel.send(replace(initial_status, message=message))

    with channel:
        try:
            with open_codex() as codex:
                if task is None:
                    models = connect(codex, show_device_code)
                    result = replace(initial_status, models=models, message="Connected to ChatGPT.")
                else:
                    with closing(AgentControlClient(get_agent_endpoint())) as client:
                        message = run_task(codex, client, task, show_progress)
                        result = replace(initial_status, message=message)
            channel.send(replace(result, phase=AgentPhase.COMPLETED))
        except Exception as error:
            channel.send(replace(initial_status, phase=AgentPhase.FAILED, message=str(error)))
