"""Explore randomized simulation episodes, synthesize a controller and test it on fresh seeds."""

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from openai_codex import Thread

from humanoid.agent.harness.episode import EpisodeFactory, open_episode
from humanoid.agent.harness.model import open_codex, start_thread
from humanoid.agent.harness.review import REVIEW_INDEX_FILENAME, ExplorationReview
from humanoid.agent.harness.worker import run_steps
from humanoid.config import ROBOT_CONFIGS, ROBOT_NAME
from humanoid.config.agent import (
    DEFAULT_AGENT_MODEL,
    TRIAL_EVALUATION_SECONDS,
    TRIAL_EXPLORATION_SECONDS,
)
from humanoid.config.api import get_api_endpoint
from humanoid.config.motion import DEFAULT_MOTION_LIMITS
from humanoid.simulation.tasks import TASKS
from humanoid.types.agent import AgentTask, EpisodeTimeExpired, TimeBudget
from humanoid.types.api import ApiEndpoint
from humanoid.types.orchestrator import Mode
from humanoid.types.process import Runtime
from humanoid.types.tools import ToolError
from humanoid.types.trial import (
    ControllerImage,
    EpisodeEnd,
    SimulationTask,
    TrialFeedback,
    TrialObservation,
    TrialResult,
    TrialStep,
    TrialSubmission,
)

IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
DRIVER = """
import json, sys
request = json.load(sys.stdin)
scope = {}
exec(request['source'], scope)
print(json.dumps(scope['step'](request['obs'], request['memory']), allow_nan=False))
"""
PROGRAM_INTERFACE = """
Submit a complete Python 3.12 standard-library program defining step(obs, memory).
It runs once per decision in an isolated environment. Return a dict with fields
command (one motion call {operation,arguments}, or None to finish) and memory (JSON object
passed to the next decision, initially {}). Use only the motion operations and limits you
explored. obs contains the same measured status fields, plus images keyed by the names
in obs["cameras"]. obs["primary_camera"] names the default camera.
Each images[name] has rgb (rows of RGB byte triples) and
registered_depth_m (axial metres aligned to RGB) and raw_depth_m (axial metres on the
depth sensor's native pixel grid). Zero means missing depth in either array. All arrays
sample every 8th pixel starting at (0,0) in their respective original frames. Use
cameras[name].raw_depth for native intrinsics and pose. During evaluation, use these
arrays directly. The NanoSAM and CDM tools are available only during exploration.
cameras[name].intrinsics uses original pixels. cameras[name].camera_pose maps
optical coordinates (right, down, forward) into
tool_pose_reference_frame. last_result is null, measured motion feedback, or {operation,error}.
seconds_remaining gives the remaining wall-clock budget. Use these live measurements
and memory to adapt to new episodes.
A fresh episode starts at home with an empty memory.
Continue until the task objective is met.
"""


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def explore(  # noqa: PLR0913 - shared agent session and episode settings
    thread: Thread,
    task: AgentTask,
    seed: int,
    output: Path,
    seconds: float,
    *,
    episodes: EpisodeFactory,
) -> None:
    messages = []

    def record(message: str) -> None:
        print(message, flush=True)
        messages.append(message)

    review_folder = output.with_suffix("")
    with ExplorationReview(review_folder, seed) as review, episodes(seed) as (episode, tools):
        episode.timer = TimeBudget(seconds)
        try:
            record(
                run_steps(
                    thread,
                    tools,
                    task,
                    record,
                    mode=Mode.AUTONOMY,
                    budget=episode.timer,
                    review=review,
                )
            )
            ended_by = EpisodeEnd.AGENT
        except EpisodeTimeExpired:
            record("Exploration time expired.")
            ended_by = EpisodeEnd.TIME_BUDGET
        elapsed_seconds = episode.timer.elapsed_seconds
        episode.release()
        score = episode.final_score()
        review.finish(ended_by.value, score)
    write_json(
        output,
        {
            **score.metrics,
            **({"lcm_recording": episode.lcm_recording} if episode.lcm_recording else {}),
            "success": score.success,
            "seed": seed,
            "messages": messages,
            "review": f"{review_folder.name}/{REVIEW_INDEX_FILENAME}",
            "elapsed_seconds": elapsed_seconds,
            "ended_by": ended_by,
            "trace": episode.trace,
        },
    )


def generate(thread: Thread, seconds: float, goal: str) -> str:
    result = thread.run(
        f"{goal}\nExploration is complete. {PROGRAM_INTERFACE}\n"
        f"Evaluation budget: {seconds} wall-clock seconds per episode. "
        f"Limits: {asdict(DEFAULT_MOTION_LIMITS)}. Return source without markdown.",
        output_schema=TrialSubmission.model_json_schema(),
    )
    if result.status.value != "completed" or not result.final_response:
        raise RuntimeError(f"Program generation ended with {result.status.value}.")
    return TrialSubmission.model_validate_json(result.final_response).source


def program_step(
    source: str, observation: TrialObservation, memory: dict[str, object], timer: TimeBudget
) -> TrialStep:
    payload = json.dumps({"source": source, "obs": observation, "memory": memory}, allow_nan=False)
    # No host files or credentials are mounted. Limit each invocation, including its output.
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65534:65534",
            "--memory=128m",
            "--pids-limit=16",
            "--ulimit",
            "fsize=65536:65536",
            "--tmpfs",
            "/tmp:size=8m",
            "--log-driver=none",
            "-e",
            "UV_CACHE_DIR=/tmp/uv",
            IMAGE,
            "sh",
            "-c",
            'timeout -s KILL 8 uv run --no-project --offline python -I -c "$1" >/tmp/result '
            "2>/tmp/error && cat /tmp/result || { cat /tmp/error >&2; exit 1; }",
            "sh",
            DRIVER,
        ],
        input=payload,
        capture_output=True,
        text=True,
        timeout=min(30, timer.remaining()),
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Controller exited {result.returncode}: {result.stderr[-2000:]}")
    return TrialStep.model_validate_json(result.stdout)


def evaluate(
    source: str,
    seed: int,
    seconds: float,
    *,
    episodes: EpisodeFactory,
    completion_reason: str,
) -> TrialResult:
    error = None
    score = None
    ended_by = EpisodeEnd.TIME_BUDGET
    memory = {}
    feedback: TrialFeedback | None = None
    with episodes(seed) as (episode, tools):
        timer = episode.timer = TimeBudget(seconds)
        try:
            while True:
                timer.remaining()
                status, views = tools.capture(Mode.AUTONOMY)
                images = {}
                for name, frames in views.items():
                    assert frames.registered is not None and frames.depth is not None
                    depth = frames.registered.image
                    images[name] = ControllerImage(
                        rgb=frames.rgb.data[::8, ::8].tolist(),
                        registered_depth_m=(depth.data[::8, ::8] * depth.depth_scale).tolist(),
                        raw_depth_m=(
                            frames.depth.data[::8, ::8] * frames.depth.depth_scale
                        ).tolist(),
                    )
                obs = TrialObservation(
                    **status,
                    seconds_remaining=timer.remaining(),
                    images=images,
                    last_result=feedback,
                )
                try:
                    decision = program_step(source, obs, memory, timer)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    timer.remaining()
                    error = str(exc)
                    ended_by = EpisodeEnd.ERROR
                    break
                timer.remaining()
                if decision.command is None:
                    ended_by = EpisodeEnd.CONTROLLER
                    break
                memory = decision.memory
                call = decision.command
                try:
                    feedback = episode.execute(
                        call.operation, call.arguments.model_dump(mode="json")
                    )
                except ValueError as exc:
                    feedback = ToolError(operation=call.operation, error=str(exc))
                if (score := episode.completed_score()) is not None:
                    ended_by = EpisodeEnd.COMPLETE
                    break
        except EpisodeTimeExpired:
            ended_by = EpisodeEnd.TIME_BUDGET
        elapsed_seconds = timer.elapsed_seconds
        episode.timer = None
        episode.release()
        if score is None:
            score = episode.final_score()
    return TrialResult(
        seed=seed,
        trace=episode.trace,
        elapsed_seconds=elapsed_seconds,
        score=score,
        lcm_recording=episode.lcm_recording,
        error=error,
        ended_by=completion_reason if ended_by is EpisodeEnd.COMPLETE else ended_by,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Runtime, choices=list(Runtime), default=Runtime.SIM)
    parser.add_argument("--endpoint", help="robot API URL for the real backend.")
    parser.add_argument("--robot", choices=ROBOT_CONFIGS, default=ROBOT_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explore", type=int, default=3)
    parser.add_argument("--test", type=int, default=3)
    parser.add_argument("--explore-seconds", type=float, default=TRIAL_EXPLORATION_SECONDS)
    parser.add_argument("--test-seconds", type=float, default=TRIAL_EVALUATION_SECONDS)
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save an LCM recording bundle for each episode.",
    )
    parser.add_argument("--replay", type=Path, help="Test a saved controller starting at --seed.")
    parser.add_argument("--output", type=Path, default=Path("logs/trials"))

    tasks = parser.add_subparsers(dest="task", required=True)
    factories = {
        name: add_arguments(tasks.add_parser(name)) for name, add_arguments in TASKS.items()
    }
    args = parser.parse_args(argv)
    if args.seed < 0 or min(args.explore, args.test) < 1:
        parser.error("Use a nonnegative seed and positive episode counts.")
    if not all(math.isfinite(s) and s > 0 for s in (args.explore_seconds, args.test_seconds)):
        parser.error("Episode budgets must be positive, finite seconds.")
    folder = args.output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    folder.mkdir(parents=True)
    print(folder.resolve(), flush=True)
    write_json(
        folder / "settings.json",
        {**vars(args), "output": str(folder), "replay": str(args.replay) if args.replay else None},
    )
    simulation_task: SimulationTask = factories[args.task](args)
    robot_config = ROBOT_CONFIGS[args.robot]
    episodes = partial(
        open_episode,
        task=simulation_task,
        robot_config=robot_config,
        backend=args.backend,
        endpoint=ApiEndpoint(args.endpoint) if args.endpoint else get_api_endpoint(),
        record=args.record,
        recording_root=folder / "recordings",
    )
    first_test_seed = args.seed
    if args.replay is not None:
        source = args.replay.read_text()
    else:
        first_test_seed += args.explore
        with open_codex() as codex:
            thread = start_thread(codex, args.model)
            for index in range(args.explore):
                instruction = (
                    f"Exploration episode {index + 1}/{args.explore} "
                    f"on the {args.backend.value} backend. "
                    f"{simulation_task.goal} "
                    "Practice the complete task before writing a reusable program. "
                    "When an attempt fails, inspect what happened, change the approach, and retry. "
                    "Carry forward strategies verified by observing the result."
                )
                task = AgentTask(instruction)
                seed = args.seed + index
                explore(
                    thread,
                    task,
                    seed,
                    folder / f"explore-{seed}.json",
                    args.explore_seconds,
                    episodes=episodes,
                )
            source = generate(thread, args.test_seconds, simulation_task.goal)
    (folder / "controller.py").write_text(source)
    errors = []
    for index in range(args.test):
        seed = first_test_seed + index
        result = evaluate(
            source,
            seed,
            args.test_seconds,
            episodes=episodes,
            completion_reason=simulation_task.completion_reason,
        )
        write_json(folder / f"test-{seed}.json", result.record())
        if result.error is not None:
            errors.append(seed)
        print(
            f"Seed {seed}: success={result.success}, {json.dumps(result.score.metrics)}, "
            f"{len(result.trace)} motions, ended_by={result.ended_by}, error={result.error}",
            flush=True,
        )

    if errors:
        raise SystemExit(f"Controller execution failed on seeds {errors}. See {folder}.")


if __name__ == "__main__":
    main()
