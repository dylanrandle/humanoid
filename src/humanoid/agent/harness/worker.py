"""One background console operation. The control service owns all robot motion."""

import signal
from collections.abc import Callable
from contextlib import closing
from multiprocessing.connection import Connection

from openai_codex import Codex, Thread

from humanoid.agent.client import AgentControlClient
from humanoid.agent.harness.model import connect, decide, open_codex, start_thread
from humanoid.agent.harness.robot import TriskelTools, image_url
from humanoid.config.agent import (
    MAX_AGENT_COMMANDS,
    MAX_AGENT_PERCEPTION_CALLS,
    get_agent_endpoint,
    get_agent_policy_config,
)
from humanoid.types.agent import AgentPhase, AgentTask, ControlOperation
from humanoid.types.agent_decision import DecisionContext, MotionArguments, ObserveAgain, ToolError
from humanoid.types.orchestrator import Mode
from humanoid.types.tools import CompleteDepth


def run_task(
    codex: Codex, client: AgentControlClient, task: AgentTask, update: Callable[[dict], None]
) -> str:
    if client.session_id != task.session_id:
        raise RuntimeError("Agent API restarted. Submit a new task.")
    with closing(TriskelTools(client)) as tools:
        thread = start_thread(codex, task.model)
        return _run_steps(thread, tools, client, task, update)


def _run_steps(
    thread: Thread,
    tools: TriskelTools,
    client: AgentControlClient,
    task: AgentTask,
    update: Callable[[dict], None],
) -> str:
    count = perception = 0
    status, image, measured_depth_image = tools.observe(task.mode)
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
            image,
            allow_motion=allow_motion,
            measured_depth_image=measured_depth_image,
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
            update({"message": f"{decision.message}\nRunning {name}."})
            if isinstance(args, ObserveAgain):
                status, image, measured_depth_image = tools.observe(task.mode)
                feedback = None
            else:
                try:
                    if isinstance(args, CompleteDepth):
                        feedback, preview = tools.depth.complete_depth(args)
                    else:
                        feedback, preview = tools.segmentation.segment(args)
                    image = image_url(preview)
                except (ValueError, OSError, RuntimeError, ImportError) as error:
                    feedback = ToolError(operation=name, error=str(error))
            continue
        if not allow_motion or client.observation().mode is not Mode.AUTONOMY:
            raise RuntimeError("Motion requires an active Autonomy task.")
        operation = ControlOperation(name)
        update({"message": f"{decision.message}\nExecuting {operation}."})
        feedback = client.execute(operation, args.model_dump(mode="json"))
        update(
            {"message": f"{operation} finished (reached={feedback['reached']}). Observing again."}
        )
        count += 1
        status, image, measured_depth_image = tools.observe(task.mode)


def _stop(signum, frame) -> None:
    # Unwind execute() and close its in-flight command before exiting.
    raise SystemExit


def work(channel: Connection, task: AgentTask | None) -> None:
    signal.signal(signal.SIGTERM, _stop)
    with channel:
        try:
            with open_codex() as codex:
                if task is None:
                    models = connect(codex, channel.send)
                    result = {"models": models, "message": "Connected to ChatGPT."}
                else:
                    with closing(AgentControlClient(get_agent_endpoint())) as client:
                        result = {"message": run_task(codex, client, task, channel.send)}
            channel.send(
                {**result, "phase": AgentPhase.COMPLETED, "login_url": "", "login_code": ""}
            )
        except Exception as error:
            channel.send(
                {
                    "phase": AgentPhase.FAILED,
                    "message": str(error),
                    "login_url": "",
                    "login_code": "",
                }
            )
