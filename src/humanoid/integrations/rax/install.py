"""Install the adapter into a separate pinned RAX checkout, without launching agents."""

import argparse
import subprocess
from pathlib import Path

RAX_REVISION = "4b577aa2ba0039ad01707ede236be9913274180e"
REGISTRY_ANCHOR = "ALL_ROBOT_TOOLS = frozenset().union"
REGISTRATION = (
    "from services.humanoid.benchmark import registration as _humanoid_registration\n"
    "BENCHMARKS['humanoid'] = _humanoid_registration(Benchmark)\n"
)


def install(checkout: Path) -> None:
    checkout = checkout.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != RAX_REVISION:
        raise ValueError(f"Expected RAX dexjoco/native revision {RAX_REVISION}, got {revision}.")
    registry = checkout / "services/benchmarks.py"
    source = registry.read_text()
    original = subprocess.check_output(
        ["git", "-C", str(checkout), "show", "HEAD:services/benchmarks.py"], text=True
    )
    expected = original.replace(REGISTRY_ANCHOR, REGISTRATION + REGISTRY_ANCHOR, 1)
    if REGISTRY_ANCHOR not in original or source not in (original, expected):
        raise ValueError(
            "RAX registry differs from the pinned version; refusing to overwrite edits."
        )
    changes = {registry: expected}
    config_path = checkout / "services/controller/config.py"
    config_original = subprocess.check_output(
        ["git", "-C", str(checkout), "show", "HEAD:services/controller/config.py"], text=True
    )
    gpu_block = config_original[config_original.index("        if not self.sim_gpu:") :]
    if not gpu_block.strip().endswith(
        """raise ValueError("Use an explicit GPU UUID, not an ordinal or 'all'")"""
    ):
        raise ValueError("Unexpected pinned RAX GPU validation block.")
    external = (
        "        if bench.name == 'humanoid':\n"
        "            if self.sim_gpu != 'external' or self.formal_sim_gpus:\n"
        "                raise ValueError('Humanoid uses the external realtime simulator')\n"
        "        else:\n"
    ) + "".join(
        "    " + line if line.strip() else line for line in gpu_block.splitlines(keepends=True)
    )
    config_expected = config_original.replace(gpu_block, external, 1)
    if config_path.read_text() not in (config_original, config_expected):
        raise ValueError(
            "RAX configuration differs from the pinned version; refusing to overwrite edits."
        )
    changes[config_path] = config_expected
    launcher = checkout / "harness/codex_cli/auto_research.py"
    launcher_original = subprocess.check_output(
        ["git", "-C", str(checkout), "show", "HEAD:harness/codex_cli/auto_research.py"],
        text=True,
    )
    anchor = "    exposed = readable(config, deployed)\n"
    helpers = (
        "    # Codex executes its filesystem helper from this private-home subdirectory.\n"
        "    # Re-expose scratch, with executable shims read-only, under the private-tree deny.\n"
        "    # Do not carve references here: that would expose private files created later.\n"
        "    sandbox_helpers = Path(deployed['codex_home'])/'tmp/arg0'\n"
        "    sandbox_helpers.mkdir(parents=True, exist_ok=True, mode=0o700)\n"
        "    exposed['write'].append(sandbox_helpers.parent)\n"
        "    exposed['read'].append(sandbox_helpers)\n"
    )
    if launcher_original.count(anchor) != 1:
        raise ValueError("Unexpected pinned RAX sandbox permission builder.")
    launcher_expected = launcher_original.replace(anchor, anchor + helpers, 1)
    if launcher.read_text() not in (launcher_original, launcher_expected):
        raise ValueError("RAX launcher has local edits; refusing to overwrite them.")
    changes[launcher] = launcher_expected
    package = checkout / "services/humanoid"
    for name in ("__init__.py", "tools.py", "backend.py", "contact_preview.py", "benchmark.py"):
        contents = (
            Path(__file__)
            .with_name(name)
            .read_text()
            .replace("from humanoid.integrations.rax.tools import", "from .tools import")
        )
        changes[package / name] = contents
    skills = Path(__file__).with_name("skills")
    for source in skills.rglob("*.md"):
        changes[checkout / "agent_skills" / source.relative_to(skills)] = source.read_text()
    # Validate every target before writing anything, including registry and config.
    for target, contents in changes.items():
        if target in (registry, config_path, launcher):
            continue
        if target.exists() and target.read_text() != contents:
            raise ValueError(f"Refusing to overwrite modified adapter {target}.")
    for target, contents in changes.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    install(args.checkout)
    print(f"Installed Humanoid adapter in {args.checkout.resolve()} (no agent started).")


if __name__ == "__main__":
    main()
