"""Replay a frozen controller using native RAX recording and bundle isolation.

Executed only inside the pinned Linux image. It never starts a model, changes the
source trial, or grants the controller access to operator files or credentials.
"""

import importlib
import json
from pathlib import Path


def main():
    runtime = Path("/home/rax/runtime")
    selection = json.loads((runtime / "replay-source.json").read_text())
    source = runtime / "operator/source-trial"
    state = json.loads((source / "state.json").read_text())
    manifest = state["bundles"][selection["bundle"]]
    if manifest["sha256"] != selection["sha256"]:
        raise ValueError("Saved controller manifest changed after replay selection.")
    frozen = (source / "bundles" / manifest["directory"]).resolve(strict=True)
    if not frozen.is_relative_to(source / "bundles") or frozen == source / "bundles":
        raise ValueError("Invalid saved controller directory.")
    storage = importlib.import_module("services.controller.storage")
    storage.verify(frozen, manifest)
    configuration = importlib.import_module("services.controller.config")
    config = configuration.Configuration.from_dict(json.loads((runtime / "input.json").read_text()))
    if any(
        state["configuration"].get(key) != getattr(config, key) for key in ("benchmark", "task")
    ):
        raise ValueError("Saved controller task does not match the replay task.")
    config.root.mkdir(parents=True, mode=0o700, exist_ok=True)
    launcher = importlib.import_module("harness.codex_cli.auto_research")
    launcher.prepare_workspace(config, "codex-home")
    backend = importlib.import_module("services.humanoid.benchmark")
    supervisor_module = importlib.import_module("services.controller.supervisor")
    supervisor = supervisor_module.Supervisor(config, lambda: backend.backend(config))
    try:
        bundle, copied = supervisor.register(frozen, workspace_path=manifest["workspace_path"])
        if copied["sha256"] != manifest["sha256"]:
            raise ValueError("Replay snapshot does not match the saved controller.")
        result = supervisor.run(bundle, formal=False)
        (runtime / "replay-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"replay_bundle": selection["bundle"], "result": result}), flush=True)
        if supervisor_module.formal_outcome(result) == "error":
            raise RuntimeError("Saved controller replay failed; inspect replay-result.json.")
    finally:
        supervisor.close()


if __name__ == "__main__":
    main()
