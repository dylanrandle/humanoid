"""Exercise the pinned Linux RAX runtime without credentials, models, or robot motion."""

import importlib
import importlib.util
import json
import os
import socket
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

# RAX is installed only in the Linux image, not in Humanoid's Python environment.
auto_research = importlib.import_module("harness.codex_cli.auto_research")
agent_host = importlib.import_module("harness.agent_host")
configuration = importlib.import_module("services.controller.config")
sandbox_module = importlib.import_module("services.controller.sandbox")

EXPLORATION_PROBE = """
import socket
from pathlib import Path
from api.runtime import Context
import numpy, scipy, PIL
with Context() as ctx:
    assert ctx.call('humanoid_status')['content'][0]['text'] == 'LOCAL_BRIDGE_OK'
print('PYTHON_MCP_BRIDGE_OK', flush=True)
try:
    Path('/home/rax/runtime/trial/new-private-file').read_text()
except (PermissionError, FileNotFoundError): pass
else: raise AssertionError('Private run file readable')
try:
    Path('/home/rax/runtime/operator/source-trial/private-source-file').read_text()
except (PermissionError, FileNotFoundError): pass
else: raise AssertionError('Private transfer source readable')
helper = Path('/home/rax/runtime/trial/codex-home/tmp/arg0/probe')
assert helper.read_text() == 'helper-dummy'
try:
    helper.write_text('changed')
except OSError: pass
else: raise AssertionError('Executable helper writable')
sock = socket.socket(socket.AF_INET)
sock.settimeout(1)
try:
    sock.connect(('1.1.1.1', 443))
except OSError: pass
else: raise AssertionError('Direct internet access available')
print('PRIVATE_FILES_AND_NETWORK_ISOLATED', flush=True)
"""

CONTROLLER_PROBE = """
def main(ctx):
    import os, socket
    from pathlib import Path
    import numpy, scipy, PIL
    assert not Path('/home/rax/runtime/trial/new-private-file').exists()
    assert not Path('/home/rax/source/services').exists()
    assert os.getenv('HUMANOID_RAX_TOKEN') is None
    assert ctx.call('humanoid_status')['content'][0]['text'] == 'LOCAL_BRIDGE_OK'
    sock = socket.socket(socket.AF_INET)
    sock.settimeout(1)
    try:
        sock.connect(('1.1.1.1', 443))
    except OSError: pass
    else: raise AssertionError('Bundle internet access available')
    print('SUBMITTED_PROGRAM_BRIDGE_OK')
"""


def respond(request):
    return {
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {"content": [{"type": "text", "text": "LOCAL_BRIDGE_OK"}]},
    }


def serve(listener):
    listener.settimeout(30)
    with listener:
        connection, _ = listener.accept()
        with connection, connection.makefile("r") as reader, connection.makefile("w") as writer:
            for line in reader:
                writer.write(json.dumps(respond(json.loads(line))) + "\n")
                writer.flush()


def main():
    config = configuration.Configuration.from_dict(json.loads(os.environ["RAX_PREFLIGHT_CONFIG"]))
    root = config.root
    root.mkdir(parents=True)
    prepared = auto_research.prepare_workspace(config, "codex-home")
    workspace = Path(prepared["workspace"])
    login = root / "agent-login.json"
    login.write_text(json.dumps({"model": "unused-runtime-probe"}))
    deployed = {"workspace": workspace, "codex_home": prepared["home"], "login": login}
    endpoint = agent_host.socket_path(root)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(endpoint))
    listener.listen()
    threading.Thread(target=serve, args=(listener,), daemon=True).start()
    auto_research.write_home(config, deployed, endpoint)
    # Create after permissions: newly created private artifacts must stay hidden too.
    (root / "new-private-file").write_text("private-dummy")
    source = root.parent / "operator/source-trial"
    source.mkdir(parents=True)
    (source / "private-source-file").write_text("private-source-dummy")
    (Path(prepared["home"]) / "tmp/arg0/probe").write_text("helper-dummy")
    environment = {
        **os.environ,
        "CODEX_HOME": str(prepared["home"]),
        "RAX_MCP_SOCKET": str(endpoint),
        "PYTHONPATH": str(workspace),
    }
    environment.pop("RAX_PREFLIGHT_CONFIG")
    subprocess.run(
        [
            "codex",
            "sandbox",
            "-P",
            "agent",
            "-C",
            str(workspace),
            "--",
            "/bin/bash",
            "-lc",
            'python -c "$RAX_RUNTIME_PROBE"',
        ],
        env={**environment, "RAX_RUNTIME_PROBE": EXPLORATION_PROBE},
        check=True,
        timeout=30,
    )
    bundle = root / "probe-bundle"
    bundle.mkdir()
    (bundle / "controller.py").write_text(CONTROLLER_PROBE)
    gateway = SimpleNamespace(
        terminal=False, simulator_exited=False, handle=lambda _, request: respond(request)
    )
    sandbox = sandbox_module.BundleSandbox(config.development)
    result = sandbox.run(
        bundle,
        mode="rehearsal",
        entrypoint="controller.py",
        gateway=gateway,
        output=root / "probe-output",
        workspace=workspace,
        wall_seconds=10,
    )
    assert result["returncode"] == 0, (root / "probe-output/stderr.log").read_text()
    assert result["output_error"] is None, result
    assert "SUBMITTED_PROGRAM_BRIDGE_OK" in (root / "probe-output/stdout.log").read_text()
    print("SUBMITTED_PROGRAM_BRIDGE_OK")
    check_replay(config)


def check_replay(config):
    """Exercise the real replay entrypoint with a native sandbox and dummy robot."""
    runtime = config.root.parent
    source = runtime / "operator/source-trial"
    program = runtime / "replay-probe-program"
    program.mkdir()
    (program / "controller.py").write_text(
        "def main(ctx):\n"
        "    import json, os\n"
        "    assert os.getenv('HUMANOID_RAX_TOKEN') is None\n"
        "    reply = ctx.call('humanoid_status')\n"
        "    assert json.loads(reply['content'][0]['text'])['active'] is True\n"
        "    print('SAVED_CONTROLLER_EXECUTED')\n"
    )
    storage = importlib.import_module("services.controller.storage")
    frozen = source / "bundles/replay-probe"
    frozen.parent.mkdir()
    manifest = storage.snapshot(
        program,
        frozen,
        max_bytes=config.formal.artifact_bytes,
        entrypoint="controller.py",
        workspace_path="code/replay-probe",
    )
    manifest["directory"] = "replay-probe"
    original_config = json.loads(os.environ["RAX_PREFLIGHT_CONFIG"])
    (source / "state.json").write_text(
        json.dumps({"configuration": original_config, "bundles": {"probe": manifest}})
    )
    selection = {"bundle": "probe", "sha256": manifest["sha256"]}
    (runtime / "replay-source.json").write_text(json.dumps(selection))
    (runtime / "input.json").write_text(
        json.dumps({**original_config, "root": str(runtime / "replay-check")})
    )
    benchmark = importlib.import_module("services.humanoid.benchmark")

    class DummyBackend:
        def tools(self):
            return benchmark.tool_definitions()

        def start(self, **_):
            return self.call("humanoid_status", {})

        def call(self, name, arguments):
            return {"active": True, "step_id": 0}, []

        def status(self):
            return self.call("humanoid_status", {})[0]

        def evaluate(self):
            return {"task_complete": False}

        def close(self):
            pass

    benchmark.__dict__["backend"] = lambda _: DummyBackend()
    spec = importlib.util.spec_from_file_location("replay_check", "/tmp/replay.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = (source / "state.json").read_bytes()
    module.main()
    result = json.loads((runtime / "replay-result.json").read_text())
    assert result["returncode"] == 0, result
    assert result["task_complete"] is False, result
    assert (source / "state.json").read_bytes() == original
    # Deliberately corrupt this disposable fixture to exercise hash verification.
    (frozen / "controller.py").chmod(0o600)
    (frozen / "controller.py").write_text("raise AssertionError('tampered')\n")
    try:
        module.main()
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("Replay accepted a modified frozen controller")
    print("SAVED_CONTROLLER_REPLAY_OK")


if __name__ == "__main__":
    main()
