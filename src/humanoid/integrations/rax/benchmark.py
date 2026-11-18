"""Import-light hooks for the pinned RAX native benchmark registry."""

from pathlib import Path

from .backend import HumanoidBackend
from .tools import tool_definitions


def registration(benchmark_type):
    names = frozenset(t["name"] for t in tool_definitions())
    return benchmark_type(
        name="humanoid",
        label="Triskel",
        adapter="services.humanoid.benchmark",
        robot_tools=names,
        geometry_tools=frozenset(),
        unscoped_tools=frozenset(),
        mutating=HumanoidBackend.mutating,
        previewable=frozenset(),
        status_tool="humanoid_status",
        observe_tool="humanoid_observe",
        tool_prefixes=("humanoid_",),
        episode_tool_prefixes=("humanoid_",),
        simulator_modules=("humanoid.integrations.rax.server",),
        startup_seconds=120,
        robot={
            "benchmark": "Triskel",
            "status": "humanoid_status",
            "observe": "humanoid_observe",
            "step": "humanoid_move_tool",
            "plan": "",
            "pixel": "",
            "hz": 50,
            "overview_camera": "rgb",
            "overview": "base-mounted",
            "cameras": ("rgb", "depth"),
            "arms": "single",
            "wrist_images": "",
            "tracking_skill": "robot-interface",
        },
        graded_score=False,
        episode_reviews=False,
        method_skills=False,
        score_label="Held-out cleanup success rate",
        layout_note="Seeded room layouts; exploration and formal use separate random namespaces.",
        setup_defaults={
            "formal_workers": 1,
            "exploration_envs": 1,
            "demonstration_context": "none",
            "expert_demonstrations": 0,
            "external_perception": "off",
        },
        cli_defaults={"formal_workers": 1, "formal_episodes": 3},
    )


def backend(config):
    validate_config(config)
    return HumanoidBackend(config)


def task_names():
    return ("toy_car_cleanup",)


def task_timeouts(task):
    if task not in task_names():
        raise ValueError(f"Unknown Humanoid task {task!r}")
    return {"episode_seconds": 1800, "startup_seconds": 120}


def validate_settings(values):
    required = {
        "formal_workers": 1,
        "exploration_envs": 1,
        "subagents": 0,
        "observation_profile": "rgbd",
        "demonstration_context": "none",
        "expert_demonstrations": 0,
    }
    for key, expected in required.items():
        if values.get(key, expected) != expected:
            raise ValueError(f"Humanoid currently requires {key}={expected!r}.")


def validate_config(config):
    task_timeouts(config.task)
    validate_settings(vars(config))
    if getattr(config, "agent_formal", False):
        raise ValueError("Humanoid integration currently evaluates submitted code bundles only.")


def validate_layouts(config):
    validate_config(config)
    for seed in (*config.exploration_seeds, config.formal_seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("Layout seeds must be nonnegative integers.")


def exploration_seeds(task, collection, start, episodes, **_):
    task_timeouts(task)
    if collection != 0 or start < 0 or episodes < 1:
        raise ValueError("Use exploration collection 0 and nonnegative layout seeds.")
    return tuple(range(start, start + episodes))


def task_context(config, demonstration, *, live_formal):
    task_timeouts(config.task)
    return (
        "# Toy car cleanup\n\n"
        "Search the entire room for toy cars. Pick up each car individually, carry it to the "
        "bin, release it inside, and repeat until all cars are collected. Pushing cars across "
        "the floor is not a substitute for carrying them. Gripper closure alone is not success. "
        "Verify possession and release from observations; report uncertainty honestly.\n\n"
        "Grasps should maintain a slight squeeze: command closure beyond estimated first "
        "contact, within the reported joint limits, rather than matching object width exactly. "
        "Keep that closing target during lifting and carrying; verify retention visually.\n\n"
        "Cars are approximately 50 x 25 mm. The bin rim is 5 inches (0.127 m) above the floor. "
        "Car height/mass and bin width are provisional in simulation. Object poses, counts, "
        "and layout seeds are not observable. Use the camera and measured robot state.\n\n"
        "This stage controls simulation only. Learn a reusable controller through RAX, "
        "rehearse it, and submit a bundle with controller.py:main(ctx). Use the supplied ctx "
        "tool API; do not depend on a simulator object or a host network endpoint. "
        "Formal evaluation uses unseen layouts.\n"
    )


def refresh_workspace(config, workspace):
    workspace = Path(workspace)
    (workspace / "TASK.md").write_text(task_context(config, None, live_formal=False))
    # The pinned upstream template installs Gemini documentation even when its
    # tool is disabled. Keep links valid without advertising an unavailable API.
    if not config.external_perception:
        gemini = workspace / ".agents/skills/gemini"
        unavailable = (
            "External perception is unavailable in this session. The generated program "
            "must use its own image processing and the supplied RGB/depth observations. "
            "The exploration agent can inspect images with its native vision.\n"
        )
        if gemini.is_dir():
            (gemini / "SKILL.md").write_text(
                "---\nname: gemini\ndescription: External perception availability.\n---\n\n"
                "# External perception\n\n" + unavailable
            )
            (gemini / "references/api.md").write_text(unavailable)
        control = workspace / ".agents/skills/autonomous-control/SKILL.md"
        if control.exists():
            control.write_text(
                control.read_text().replace(
                    "- Use Gemini when simple perception rules are insufficient. Write your own\n"
                    "  prompts and response-processing code; "
                    "see the [gemini skill](../gemini/SKILL.md).",
                    "- Use image processing and native depth in the generated program. "
                    "External perception is unavailable in this session.",
                )
            )


def reference_mounts(config):
    return []


def fetch_demonstration(demos_root, task, kind):
    task_timeouts(task)
    if kind != "none":
        raise ValueError("No demonstrations are supplied for this task.")


def provision_demonstration(config, target, demos_root):
    fetch_demonstration(demos_root, config.task, config.demonstration_context)
    return {
        "kind": "none",
        "image_count": 0,
        "root": "runtime/demonstrations",
        "manifest_path": None,
    }


def episode_reviewer(root):
    return None


def review_analysis():
    return None


def unscoped_call(name, arguments):
    raise ValueError(f"No unscoped Humanoid tool {name!r}.")
