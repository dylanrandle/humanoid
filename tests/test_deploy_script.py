import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = PROJECT_ROOT / "scripts" / "deploy"
INVALID_ARGUMENT_EXIT_CODE = 2


@pytest.mark.parametrize(
    "install_dir",
    [
        "/",
        "//",
        "/opt/..",
        "/opt/./humanoid",
        "/opt//humanoid",
        "/opt/humanoid/",
    ],
)
def test_deploy_rejects_non_normalized_install_directories(install_dir: str) -> None:
    environment = {**os.environ, "HUMANOID_ROBOT_DIR": install_dir}

    result = subprocess.run(
        [DEPLOY_SCRIPT, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == INVALID_ARGUMENT_EXIT_CODE
    assert "must be a normalized absolute path" in result.stderr
    assert "Building Humanoid" not in result.stdout
