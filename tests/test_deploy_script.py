import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from humanoid.constants import EnvironmentVariable, Host, Port

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = PROJECT_ROOT / "scripts" / "deploy"
INVALID_ARGUMENT_EXIT_CODE = 2
PRIVATE_FILE_MODE = 0o600
PRIVATE_DIRECTORY_MODE = 0o700


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
    environment = {**os.environ, EnvironmentVariable.ROBOT_DIR: install_dir}

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


def test_deploy_dry_run_packages_https_without_contacting_robot(tmp_path: Path):
    script = tmp_path / "scripts/deploy"
    script.parent.mkdir()
    shutil.copy2(DEPLOY_SCRIPT, script)
    shutil.copytree(PROJECT_ROOT / "deploy", tmp_path / "deploy")
    shutil.copy2(PROJECT_ROOT / ".python-version", tmp_path / ".python-version")
    for relative in ("cdm/d405", "cdm/d435", "nanosam/encoder", "nanosam/decoder"):
        engine = tmp_path / "models" / f"{relative}.plan"
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"fixture engine")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text(
        "#!/bin/bash\nset -eu\n"
        'if [[ "$1" == build ]]; then\n'
        '  touch "$4/humanoid-test.whl"\n'
        'elif [[ "$1" == run ]]; then\n'
        '  exec "$DEPLOY_TEST_UV" run --project "$DEPLOY_TEST_PROJECT" "${@:2}"\n'
        "else\n"
        '  for arg in "$@"; do destination="$arg"; done\n'
        '  touch "$destination"\n'
        "fi\n"
    )
    uv.chmod(0o755)
    # Record the actual archive contents before the deploy script cleans up its bundle.
    tar = binaries / "tar"
    tar.write_text(
        "#!/bin/bash\nset -eu\n"
        '"$DEPLOY_TEST_TAR" "$@"\n'
        '"$DEPLOY_TEST_TAR" -tzf "$5" > "$DEPLOY_TEST_CAPTURE/bundle-files"\n'
        '"$DEPLOY_TEST_TAR" -xOf "$5" Caddyfile > "$DEPLOY_TEST_CAPTURE/Caddyfile"\n'
    )
    tar.chmod(0o755)
    for name in ("ssh", "scp", "rsync"):
        forbidden = binaries / name
        forbidden.write_text('#!/bin/bash\ntouch "$DEPLOY_TEST_CAPTURE/contacted-robot"\nexit 99\n')
        forbidden.chmod(0o755)
    result = subprocess.run(
        [script, "--dry-run"],
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "DEPLOY_TEST_CAPTURE": str(tmp_path),
            "DEPLOY_TEST_TAR": shutil.which("tar") or "tar",
            "DEPLOY_TEST_UV": shutil.which("uv") or "uv",
            "DEPLOY_TEST_PROJECT": str(PROJECT_ROOT),
            EnvironmentVariable.QUEST_PORT: "9101",
            EnvironmentVariable.DASHBOARD_PORT: "9102",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "contacted-robot").exists()
    assert {"Caddyfile", "humanoid-gateway.service.template", "humanoid.service.template"} <= set(
        (tmp_path / "bundle-files").read_text().splitlines()
    )
    assert "restart humanoid.service and humanoid-gateway.service" in result.stdout
    rendered = (tmp_path / "Caddyfile").read_text()
    assert f"https://{{$HUMANOID_PUBLIC_HOST}}:{Port.QUEST_GATEWAY}" in rendered
    assert f"https://{{$HUMANOID_PUBLIC_HOST}}:{Port.DASHBOARD_GATEWAY}" in rendered
    assert f"reverse_proxy {Host.LOOPBACK}:{Port.API}" in rendered
    assert f"reverse_proxy {Host.LOOPBACK}:{Port.QUEST_SERVER}" in rendered
    assert "@" not in rendered
    assert "9101" not in rendered and "9102" not in rendered


@pytest.mark.parametrize(
    ("machine", "valid_checksum", "preinstalled", "succeeds"),
    [
        ("aarch64", True, False, True),
        ("x86_64", True, False, True),
        ("aarch64", False, False, False),
        ("unsupported", True, False, False),
        ("unsupported", False, True, True),
    ],
)
def test_caddy_bootstrap_verifies_download_and_preserves_existing_binary(
    tmp_path: Path, machine: str, valid_checksum: bool, preinstalled: bool, succeeds: bool
):
    # Exercise the exact remote bootstrap without SSH, sudo, or network access.
    source = DEPLOY_SCRIPT.read_text()
    bootstrap = source[source.index("install_caddy() (") : source.index("\ninstall_caddy\n")]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    installed = tmp_path / "caddy"
    if preinstalled:
        installed.write_bytes(b"existing caddy")
        installed.chmod(0o755)
    executable = tmp_path / "fixture/caddy"
    executable.parent.mkdir()
    executable.write_bytes(b"verified caddy")
    for arch in ("arm64", "amd64"):
        archive = tmp_path / f"caddy_2.11.9_linux_{arch}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(executable, arcname="caddy")
        checksum = hashlib.sha512(archive.read_bytes()).hexdigest() if valid_checksum else "0" * 128
        with (tmp_path / "caddy_2.11.9_checksums.txt").open("a") as checksums:
            checksums.write(f"{checksum}  {archive.name}\n")
    mocks = {
        "uname": f"echo {machine}\n",
        "curl": 'cp "$DEPLOY_TEST_FIXTURES/${4##*/}" "$6"\n',
    }
    if not shutil.which("sha512sum"):
        mocks["sha512sum"] = 'exec shasum -a 512 "$@"\n'
    for name, body in mocks.items():
        command = binaries / name
        command.write_text("#!/bin/bash\nset -eu\n" + body)
        command.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + bootstrap + "\ninstall_caddy"],
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "CADDY": str(installed),
            "DEPLOY_TEST_FIXTURES": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is succeeds, result.stderr
    if succeeds:
        assert installed.read_bytes() == (b"existing caddy" if preinstalled else b"verified caddy")
    else:
        assert not installed.exists()


def test_gateway_configuration_uses_install_directory_and_preserves_ca(tmp_path: Path):
    source = DEPLOY_SCRIPT.read_text()
    start = source.index('install -d -m 0755 "$GATEWAY_DIR"')
    configuration = source[start : source.index("\nPYTHON_VERSION=", start)]
    gateway = tmp_path / "custom-install/gateway"
    authority = gateway / "data/caddy/pki/authorities/local"
    authority.mkdir(parents=True)
    certificate = authority / "root.crt"
    key = authority / "root.key"
    certificate.write_bytes(b"existing CA")
    key.write_bytes(b"existing private key")
    key.chmod(PRIVATE_FILE_MODE)
    staging = tmp_path / "staging"
    staging.mkdir()
    shutil.copy2(PROJECT_ROOT / "deploy/Caddyfile", staging / "Caddyfile")
    caddy = tmp_path / "caddy"
    caddy.write_text(
        "#!/bin/bash\nset -eu\n"
        '[[ "$1" == validate ]]\n'
        '[[ "$XDG_DATA_HOME" == "$GATEWAY_DIR/data" ]]\n'
        '[[ "$XDG_CONFIG_HOME" == "$GATEWAY_DIR/config" ]]\n'
        '[[ "$HUMANOID_PUBLIC_HOST" == robot.local ]]\n'
    )
    caddy.chmod(0o755)
    for _ in range(2):
        subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + configuration],
            env={
                **os.environ,
                "CADDY": str(caddy),
                "GATEWAY_DIR": str(gateway),
                "STAGING_DIR": str(staging),
                "PUBLIC_HOST": "robot.local",
            },
            check=True,
            capture_output=True,
            text=True,
        )
        assert certificate.read_bytes() == b"existing CA"
        assert key.read_bytes() == b"existing private key"
        assert key.stat().st_mode & 0o777 == PRIVATE_FILE_MODE
        for directory in (gateway / "data", gateway / "config"):
            assert directory.stat().st_mode & 0o777 == PRIVATE_DIRECTORY_MODE
        assert (gateway / "Caddyfile").read_bytes() == (staging / "Caddyfile").read_bytes()
