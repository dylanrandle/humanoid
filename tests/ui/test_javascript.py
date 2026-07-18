import subprocess
from pathlib import Path

from humanoid.utils.paths import find_repo_root


def test_frontend_behavior_suite():
    project_root = find_repo_root(__file__)
    tests = sorted((Path(__file__).with_name("js")).glob("*.test.js"))

    subprocess.run(
        ["node", "--test", *(str(test) for test in tests)],
        cwd=project_root,
        check=True,
    )
