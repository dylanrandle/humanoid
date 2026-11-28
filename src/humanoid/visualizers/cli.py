"""View live robot telemetry for a running hardware or simulation stack."""

import argparse
import os
import sys
from pathlib import Path

from humanoid.constants import Host
from humanoid.visualizers.native import show
from humanoid.visualizers.stream import local_frames, ssh_frames


def run(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="Read LCM over SSH from HOST. Defaults to the local bus.")
    args = parser.parse_args(argv)

    try:
        if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
            launcher = Path(sys.executable).with_name("mjpython")
            os.execv(launcher, [str(launcher), "-m", "humanoid.visualizers.cli", *argv])
        with ssh_frames(args.host) if args.host else local_frames() as receive:
            show(receive, args.host or Host.LOCALHOST)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(str(exc))


def main() -> None:
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
