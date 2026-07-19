"""Run the complete project verification suite."""

import subprocess

from humanoid.utils.paths import find_repo_root

CHECK_COMMANDS = (
    ("ruff", "format", "--check"),
    ("npm", "run", "format:check"),
    ("ruff", "check"),
    ("ty", "check"),
    ("pytest",),
)


def main() -> None:
    project_root = find_repo_root(__file__)
    for command in CHECK_COMMANDS:
        print(f"\n> {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=project_root, check=True)
