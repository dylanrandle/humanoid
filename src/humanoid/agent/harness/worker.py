"""One background console operation. The control service owns all robot motion."""

import signal
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from multiprocessing.connection import Connection

from openai_codex import Codex, Thread

from humanoid.agent.harness.model import connect, decide, open_codex, start_thread
from humanoid.agent.harness.review import ExplorationReview
from humanoid.agent.harness.robot import RobotTools, image_url
from humanoid.api.clients.robot import RobotClient
from humanoid.config.agent import CONSOLE_TASK_SECONDS
from humanoid.config.api import get_api_endpoint
from humanoid.config.motion import DEFAULT_MOTION_LIMITS, DEFAULT_MOTION_TIMEOUT_SECONDS
from humanoid.config.robot import ROBOT_CONFIGS
from humanoid.robots.base import Robot
from humanoid.types.agent import (
    AgentExecution,
    AgentPanelStatus,
    AgentPhase,
    AgentTask,
    DecisionContext,
    TimeBudget,
)
from humanoid.types.motion import ControlOperation, MotionArguments, MotionResult
from humanoid.types.orchestrator import Mode
from humanoid.types.segmentation import SegmentationResult
from humanoid.types.tools import (
    CompleteDepth,
    DepthCompletionResult,
    DepthSamples,
    ObserveAgain,
    ReadDepth,
    SegmentRequest,
    ToolError,
)


def run_execution(
    codex: Codex, client: RobotClient, execution: AgentExecution, update: Callable[[str], None]
) -> str:
    budget = TimeBudget(CONSOLE_TASK_SECONDS)
    if client.runtime_id != execution.runtime_id:
        raise RuntimeError("Robot runtime restarted. Submit a new task.")
    if execution.mode is Mode.AUTONOMY and client.motion_session_id != execution.motion_session_id:
        raise RuntimeError("Motion executor restarted. Submit a new task.")
    thread = start_thread(codex, execution.model)
    with closing(RobotTools(client, Robot(ROBOT_CONFIGS[execution.robot]))) as tools:
        return run_steps(thread, tools, execution.task, update, mode=execution.mode, budget=budget)


def _perceive(
    tools: RobotTools, args: ReadDepth | CompleteDepth | SegmentRequest, images: dict[str, str]
) -> DepthSamples | DepthCompletionResult | SegmentationResult:
    if isinstance(args, ReadDepth):
        return tools.depth.read_depth(args)
    if isinstance(args, CompleteDepth):
        return tools.depth.complete_depth(args)
    result, preview = tools.segmentation.segment(args)
    images["Perception result for " + result.observation_id] = image_url(preview)
    return result


def _move(
    tools: RobotTools,
    operation: ControlOperation,
    args: MotionArguments,
    budget: TimeBudget,
    update: Callable[[str], None],
) -> MotionResult | ToolError:
    try:
        feedback = tools.client.execute(
            operation,
            args.model_dump(mode="json"),
            timeout=min(DEFAULT_MOTION_TIMEOUT_SECONDS, budget.remaining()),
        )
        update(f"{operation} finished (reached={feedback['reached']}). Observing again.")
        return feedback
    except ValueError as error:
        update(f"{operation} rejected: {error}")
        return ToolError(operation=operation, error=str(error))
    except RuntimeError:
        budget.remaining()
        raise


def run_steps(  # noqa: PLR0913 - the optional review observes the same live decision loop.
    thread: Thread,
    tools: RobotTools,
    task: AgentTask,
    update: Callable[[str], None],
    *,
    mode: Mode,
    budget: TimeBudget,
    review: ExplorationReview | None = None,
) -> str:
    allow_motion = mode is Mode.AUTONOMY
    budget.remaining()
    status, images = tools.observe(mode)
    feedback = None
    while True:
        context = DecisionContext(
            instruction=task.instruction,
            observation=status,
            last_result=feedback,
            seconds_remaining=budget.remaining(),
            limits=DEFAULT_MOTION_LIMITS,
        )
        if review:
            review.observe(budget.elapsed_seconds, context, images)
        decision = decide(
            thread,
            context,
            images,
            allow_motion=allow_motion,
            timer=budget,
        )
        if review:
            review.decided(decision)
        budget.remaining()
        command = decision.command
        if command is None:
            return decision.message
        name = command.operation
        args = command.arguments
        if not isinstance(args, MotionArguments):
            if tools.client.observation().mode is not mode:
                raise RuntimeError("Robot mode changed during the task.")
            update(f"{decision.message}\nRunning {name}.")
            budget.remaining()
            if isinstance(args, ObserveAgain):
                status, images = tools.observe(mode)
                feedback = None
            else:
                try:
                    feedback = _perceive(tools, args, images)
                except (ValueError, OSError, RuntimeError, ImportError) as error:
                    feedback = ToolError(operation=name, error=str(error))
            if review:
                preview = (
                    images.get("Perception result for " + feedback.observation_id)
                    if isinstance(feedback, SegmentationResult)
                    else None
                )
                review.outcome(feedback, preview)
            continue
        if not allow_motion or tools.client.observation().mode is not Mode.AUTONOMY:
            raise RuntimeError("Motion requires an active API task.")
        operation = ControlOperation(name)
        update(f"{decision.message}\nExecuting {operation}.")
        feedback = _move(tools, operation, args, budget, update)
        if review:
            review.outcome(feedback)
        budget.remaining()
        status, images = tools.observe(mode)


def _stop(signum, frame) -> None:
    # Unwind execute() and close its in-flight command before exiting.
    raise SystemExit


def work(
    channel: Connection, execution: AgentExecution | None, initial_status: AgentPanelStatus
) -> None:
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
                if execution is None:
                    models = connect(codex, show_device_code)
                    result = replace(initial_status, models=models, message="Connected to ChatGPT.")
                else:
                    with closing(RobotClient(get_api_endpoint())) as client:
                        message = run_execution(codex, client, execution, show_progress)
                        result = replace(initial_status, message=message)
            channel.send(replace(result, phase=AgentPhase.COMPLETED))
        except Exception as error:
            channel.send(replace(initial_status, phase=AgentPhase.FAILED, message=str(error)))
