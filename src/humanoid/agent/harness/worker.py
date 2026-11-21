"""One background console operation. The control service owns all robot motion."""

import signal
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict
from multiprocessing.connection import Connection

from openai_codex import Codex

from humanoid.agent.client import AgentControlClient
from humanoid.agent.harness.model import connect, decide, open_codex, start_thread
from humanoid.agent.harness.robot import TriskelTools
from humanoid.config.agent import MAX_AGENT_COMMANDS, get_agent_endpoint, get_agent_policy_config
from humanoid.types.agent import AgentPhase, AgentTask, ControlOperation
from humanoid.types.orchestrator import Mode


def run_task(
    codex: Codex, client: AgentControlClient, task: AgentTask, update: Callable[[dict], None]
) -> str:
    if client.session_id != task.session_id:
        raise RuntimeError("Agent API restarted. Submit a new task.")
    tools = TriskelTools(client)
    thread = start_thread(codex, task.model)
    feedback = None
    for count in range(MAX_AGENT_COMMANDS + 1):
        status, image = tools.observe(task.mode)
        allow_motion = task.mode is Mode.AUTONOMY and count < MAX_AGENT_COMMANDS
        context = {
            "instruction": task.instruction,
            "observation": status,
            "last_result": feedback,
            "commands_remaining": MAX_AGENT_COMMANDS - count if allow_motion else 0,
            "limits": asdict(get_agent_policy_config()),
        }
        decision = decide(thread, context, image, allow_motion=allow_motion)
        command = decision.get("command")
        if command is None:
            return decision["message"]
        if not allow_motion or client.observation().mode is not Mode.AUTONOMY:
            raise RuntimeError("Motion requires an active Autonomy task.")
        operation = ControlOperation(command["operation"])
        update({"message": f"{decision['message']}\nExecuting {operation}."})
        feedback = client.execute(operation, command["arguments"])
        update(
            {"message": f"{operation} finished (reached={feedback['reached']}). Observing again."}
        )
    raise RuntimeError("Agent command budget exhausted.")


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
