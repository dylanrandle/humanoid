import os
import shutil
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


@pytest.mark.parametrize("contents", [None, b"", b"version https://git-lfs.github.com/spec/v1\n"])
def test_deploy_requires_materialized_engines_before_build(tmp_path: Path, contents: bytes | None):
    script = tmp_path / "scripts" / "deploy"
    script.parent.mkdir()
    shutil.copy2(DEPLOY_SCRIPT, script)
    for relative in ("cdm/d405", "cdm/d435", "nanosam/encoder"):
        engine = tmp_path / "models" / f"{relative}.plan"
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"fixture engine")
    if contents is not None:
        (tmp_path / "models/nanosam/decoder.plan").write_bytes(contents)

    result = subprocess.run([script, "--dry-run"], capture_output=True, text=True, check=False)

    assert result.returncode == 1
    assert "models/nanosam/decoder.plan" in result.stderr
    assert "git lfs pull" in result.stderr
    assert "Building Humanoid" not in result.stdout
