"""Local Docker lifecycle for the pinned native RAX launcher."""

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from functools import cache
from pathlib import Path

from humanoid.config.rax import (
    RAX_EXPLORATION_SEEDS,
    RAX_FORMAL_EPISODES,
    RAX_FORMAL_SEED,
    RAX_IMAGE,
    RAX_LIMITS,
    RAX_MODEL,
)
from humanoid.integrations.rax.install import RAX_REVISION, install
from humanoid.integrations.rax.tools import CONTACT_PREVIEW_TASK
from humanoid.utils.paths import find_data_root, find_repo_root


def assets() -> Path:
    return find_repo_root(__file__) / "deploy/rax"


def auth_path() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json"


@cache
def default_command() -> list[str]:
    """Resolve installed runner prerequisites once when the UI opens."""
    docker = shutil.which("docker")
    if sys.platform != "darwin" or docker is None or not auth_path().is_file():
        return []
    try:
        assets()
        available = (
            subprocess.run(
                [docker, "image", "inspect", RAX_IMAGE],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if not available:
        return []
    return [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        "-m",
        "humanoid.integrations.rax.runner",
    ]


def configuration(task: str) -> dict:
    return {
        "root": "/home/rax/runtime/trial",
        "task": task,
        "benchmark": "humanoid",
        "sim_gpu": "external",
        "exploration_seeds": [0, 1]
        if task == CONTACT_PREVIEW_TASK
        else list(RAX_EXPLORATION_SEEDS),
        "formal_seed": RAX_FORMAL_SEED,
        "eval_seed": 0,
        "formal_episodes": 1 if task == CONTACT_PREVIEW_TASK else RAX_FORMAL_EPISODES,
        "formal_workers": 1,
        "exploration_envs": 1,
        "subagents": 0,
        "observation_profile": "rgbd",
        "demonstration_context": "none",
        "expert_demonstrations": 0,
        "external_perception": False,
        "agent_cli": "codex",
        "agent_provider": "custom",
        "development": dict(RAX_LIMITS),
        "formal": dict(RAX_LIMITS),
        "training": dict(RAX_LIMITS),
    }


def prepare_run(directory: Path, task: str, *, adapt_from: Path | None = None) -> None:
    config = configuration(task)
    source = None
    if adapt_from is not None:
        source = adapt_from.resolve(strict=True) / "trial"
        previous = json.loads((source / "state.json").read_text())
        original = previous["configuration"]
        if original.get("benchmark") != "humanoid" or original.get("task") != task:
            raise ValueError("The source trial must match the selected Humanoid task.")
        if previous.get("active") or any(
            (previous.get(key) or {}).get("status") == "running"
            for key in ("formal_batch", "agent_formal_batch")
        ):
            raise ValueError("Finish the source trial's active episode and evaluation first.")
        # Native RAX also takes the source locks before transferring anything.
        # Keep source history immutable and use new exploration layouts.
        held_out = set(
            range(config["formal_seed"], config["formal_seed"] + config["formal_episodes"])
        )
        seed = max(original["exploration_seeds"]) + 1
        seeds = []
        while len(seeds) < len(config["exploration_seeds"]):
            if seed not in held_out:
                seeds.append(seed)
            seed += 1
        config.update(
            exploration_seeds=seeds,
            adaptation_source="/home/rax/runtime/operator/source-trial",
            adaptation_context="compact",
        )
    directory.mkdir(parents=True, mode=0o700)
    operator = directory / "operator/.codex"
    operator.mkdir(parents=True, mode=0o700)
    (operator / "auth.json").touch(mode=0o600)  # Mount point; no credential copy.
    (operator / "chatgpt.config.toml").write_text(
        f'model = {json.dumps(RAX_MODEL)}\n[rax]\nchatgpt_auth = "auth.json"\n'
    )
    (directory / "input.json").write_text(json.dumps(config, indent=2) + "\n")
    if source is not None:
        (operator.parent / "source-trial").mkdir()
        (directory / "transfer-source.json").write_text(json.dumps({"trial": str(source)}) + "\n")


def docker_command(
    directory: Path, name: str, *, prepare_only: bool, resume: bool = False
) -> list[str]:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker Desktop is required for the local RAX runner.")
    replay = directory / "replay-source.json"
    if not replay.is_file() and not auth_path().is_file():
        raise RuntimeError("Log in to Codex on this Mac before starting RAX.")
    command = [
        *container_command(docker),
        "--name",
        name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,source={directory},target=/home/rax/runtime",
        "--env",
        "HOME=/home/rax/runtime/operator",
    ]
    task = json.loads((directory / "input.json").read_text())["task"]
    command += ["--env", f"HUMANOID_RAX_TASK={task}"]
    if task == CONTACT_PREVIEW_TASK:
        command += ["--env", "HUMANOID_OBSERVATION_ENDPOINT"]
    else:
        command += [
            "--env",
            "HUMANOID_RAX_TOKEN",
            "--env",
            "HUMANOID_RAX_ENDPOINT="
            + os.environ.get(
                "HUMANOID_RAX_ENDPOINT", "http://host.docker.internal:8765/api/autonomy/rpc"
            ),
        ]
    if replay.is_file():
        selection = json.loads(replay.read_text())
        source = Path(selection["trial"]).resolve(strict=True)
        return [
            *command,
            "--mount",
            f"type=bind,source={source},target=/home/rax/runtime/operator/source-trial,readonly",
            "--mount",
            f"type=bind,source={assets() / 'replay.py'},target=/tmp/replay.py,readonly",
            RAX_IMAGE,
            "/tmp/replay.py",
        ]
    command += [
        "--mount",
        f"type=bind,source={auth_path()},target=/home/rax/runtime/operator/.codex/auth.json,readonly",
    ]
    transfer = directory / "transfer-source.json"
    if transfer.is_file():
        source = Path(json.loads(transfer.read_text())["trial"]).resolve(strict=True)
        command += [
            "--mount",
            f"type=bind,source={source},target=/home/rax/runtime/operator/source-trial,readonly",
        ]
    command += [
        RAX_IMAGE,
        "-m",
        "harness.codex_cli.auto_research",
        "--config",
        "/home/rax/runtime/input.json",
        "--provider-home",
        "/home/rax/runtime/operator/.codex",
        "--profile",
        "chatgpt",
        "--prepare-only" if prepare_only else "--non-interactive",
    ]
    if resume:
        command.append("--resume")
    return command


def container_command(docker: str) -> list[str]:
    """Identical container isolation for validation and actual exploration."""
    return [
        docker,
        "run",
        "--rm",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--security-opt",
        f"seccomp={assets() / 'seccomp.json'}",
        "--security-opt",
        "systempaths=unconfined",
        "--memory",
        "5g",
        "--cpus",
        "2",
        "--pids-limit",
        "512",
    ]


def check_runtime() -> None:
    """Check real sandbox transports with dummy data, no login, model, or robot."""
    subprocess.run(
        [
            *container_command("docker"),
            "--env",
            "PYTHONPATH=/home/rax/source",
            "--env",
            "RAX_PREFLIGHT_CONFIG=" + json.dumps(configuration("toy_car_cleanup")),
            "--mount",
            f"type=bind,source={assets() / 'check_runtime.py'},"
            "target=/tmp/check_runtime.py,readonly",
            "--mount",
            f"type=bind,source={assets() / 'replay.py'},target=/tmp/replay.py,readonly",
            RAX_IMAGE,
            "/tmp/check_runtime.py",
        ],
        check=True,
        timeout=60,
    )


def run(
    *, prepare_only: bool = False, resume: Path | None = None, adapt_from: Path | None = None
) -> int:
    if resume is not None and adapt_from is not None:
        raise ValueError("Choose either resume or adaptation into a new trial.")
    task = os.environ.get("HUMANOID_RAX_TASK", "toy_car_cleanup")
    name = "humanoid-rax-" + uuid.uuid4().hex[:12]
    if resume is None:
        directory = find_data_root(__file__) / "logs/rax" / name
        prepare_run(directory, task, adapt_from=adapt_from)
    else:
        if prepare_only:
            raise ValueError("A prepared-only check cannot resume an exploration run.")
        directory = resume.resolve(strict=True)
        config = json.loads((directory / "input.json").read_text())
        if config.get("benchmark") != "humanoid" or config.get("task") != task:
            raise ValueError("The resumed run must match the selected Humanoid task.")
        if not (directory / "trial/agent-lifecycle.json").is_file():
            raise ValueError("The selected directory has no launched RAX session to resume.")
    # A fresh container name avoids accidentally removing an existing run if
    # RAX rejects a concurrent resume while holding the session's launch lock.
    command = docker_command(directory, name, prepare_only=prepare_only, resume=resume is not None)
    return _launch(directory, name, command)


def prepare_replay(
    directory: Path, task: str, source: Path, *, bundle: str | None = None, seed: int = 0
) -> None:
    """Prepare a separate simulation run of an immutable native RAX bundle."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Replay seed must be a nonnegative integer.")
    trial = source.resolve(strict=True) / "trial"
    state = json.loads((trial / "state.json").read_text())
    original = state["configuration"]
    if original.get("benchmark") != "humanoid" or original.get("task") != task:
        raise ValueError("The saved controller must match the selected Humanoid task.")
    if state.get("active") or any(
        (state.get(key) or {}).get("status") == "running"
        for key in ("formal_batch", "agent_formal_batch")
    ):
        raise ValueError("Wait for the source run's episode and evaluation to finish.")
    bundles = state.get("bundles", {})
    bundle = bundle or (state.get("submission") or {}).get("bundle")
    if bundle is None and len(bundles) == 1:
        bundle = next(iter(bundles))
    if bundle not in bundles:
        raise ValueError("Choose a registered saved controller with --bundle.")
    manifest = bundles[bundle]
    bundle_root = (trial / "bundles").resolve(strict=True)
    frozen = (bundle_root / manifest["directory"]).resolve(strict=True)
    if not frozen.is_dir() or not frozen.is_relative_to(bundle_root) or frozen == bundle_root:
        raise ValueError("Saved controller directory must be inside the source bundle store.")
    config = configuration(task)
    config["exploration_seeds"] = [seed]
    directory.mkdir(parents=True, mode=0o700)
    (directory / "operator/source-trial").mkdir(parents=True)
    (directory / "input.json").write_text(json.dumps(config, indent=2) + "\n")
    (directory / "replay-source.json").write_text(
        json.dumps({"trial": str(trial), "bundle": bundle, "sha256": manifest["sha256"]}) + "\n"
    )


def replay_run(source: Path, *, bundle: str | None = None, seed: int = 0) -> int:
    task = os.environ.get("HUMANOID_RAX_TASK", "toy_car_cleanup")
    name = "humanoid-replay-" + uuid.uuid4().hex[:12]
    directory = find_data_root(__file__) / "logs/rax" / name
    prepare_replay(directory, task, source, bundle=bundle, seed=seed)
    return _launch(directory, name, docker_command(directory, name, prepare_only=False))


def _launch(directory: Path, name: str, command: list[str]) -> int:
    print(f"RAX artifacts: {directory}", flush=True)

    def interrupt(*_):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        # Stop the container even if the attached Docker client was interrupted.
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                [command[0], "rm", "--force", name],
                timeout=1,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def build() -> None:
    """Build from the verified upstream revision and its pinned analysis dependencies."""
    with tempfile.TemporaryDirectory(prefix="humanoid-rax-build-") as temporary:
        context = Path(temporary)
        checkout = context / "rax"
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--single-branch",
                "--branch",
                "dexjoco/native",
                "https://github.com/robot-agent-exploration/rax",
                str(checkout),
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "checkout", RAX_REVISION], check=True)
        install(checkout)
        shutil.copyfile(assets() / "Dockerfile", context / "Dockerfile")
        packages = {
            "NUMPY": "numpy",
            "PILLOW": "pillow",
            "JINJA2": "jinja2",
            "MSGPACK": "msgpack",
            "PYYAML": "pyyaml",
            "IMAGEIO": "imageio",
            "IMAGEIO_FFMPEG": "imageio-ffmpeg",
            "SCIPY": "scipy",
            "PYTEST": "pytest",
            "OPENCV": "opencv-python-headless",
        }
        pins = dict(
            line.split("=", 1)
            for line in (checkout / "dependencies.lock").read_text().splitlines()
            if line and not line.startswith("#")
        )
        requirements = [
            f"{package}=={pins[f'HOST_{key}_VERSION']}" for key, package in packages.items()
        ]
        requirements.append(f"h5py=={pins['AGENT_H5PY_VERSION']}")
        (context / "requirements.txt").write_text("\n".join(requirements) + "\n")
        subprocess.run(["docker", "build", "--tag", RAX_IMAGE, str(context)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build the pinned Linux runner image")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Validate deployment without model or episode"
    )
    parser.add_argument(
        "--check-runtime", action="store_true", help="Check sandbox and Python tool transport"
    )
    parser.add_argument("--resume", type=Path, help="Resume an existing run artifact directory")
    parser.add_argument(
        "--adapt-from", type=Path, help="Transfer a finished run into a new RAX trial"
    )
    parser.add_argument(
        "--replay-from",
        type=Path,
        help="Run a saved controller for the selected task without a model",
    )
    parser.add_argument("--bundle", help="Registered bundle ID to replay; defaults to submission")
    parser.add_argument("--seed", type=int, default=0, help="Simulation layout seed for replay")
    parser.add_argument(
        "--contact-preview",
        action="store_true",
        help="Run the read-only toy car contact trial against the forwarded hardware dashboard",
    )
    args = parser.parse_args()
    if args.contact_preview:
        os.environ["HUMANOID_RAX_TASK"] = CONTACT_PREVIEW_TASK
    if args.replay_from and (args.resume or args.adapt_from or args.prepare_only):
        parser.error("Replay cannot be combined with resume, adaptation, or prepare-only.")
    if (args.bundle is not None or args.seed != 0) and args.replay_from is None:
        parser.error("--bundle and --seed require --replay-from.")
    if args.build:
        build()
    elif args.check_runtime:
        check_runtime()
    elif args.replay_from:
        raise SystemExit(replay_run(args.replay_from, bundle=args.bundle, seed=args.seed))
    else:
        raise SystemExit(
            run(prepare_only=args.prepare_only, resume=args.resume, adapt_from=args.adapt_from)
        )


if __name__ == "__main__":
    main()
