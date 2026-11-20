"""Keep prepared model environments on the deployed Humanoid package revision."""

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path

from humanoid.config.deployment import MODEL_PYTHONS


def install_worker_package(wheel: Path, pythons: Sequence[Path], uv: str) -> None:
    """Update only Humanoid, preserving vendor CUDA packages and model artifacts."""
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError(f"Expected a Humanoid wheel: {wheel}")
    interpreters = {python.expanduser() for python in pythons}
    for python in sorted(interpreters):
        if not python.is_file():
            print(f"Model environment not prepared; skipping {python}", flush=True)
            continue
        print(f"Updating Humanoid in model environment: {python}", flush=True)
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "--reinstall-package",
                "humanoid",
                str(wheel.resolve()),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--uv", required=True)
    args = parser.parse_args()
    install_worker_package(args.wheel, MODEL_PYTHONS, args.uv)


if __name__ == "__main__":
    main()
