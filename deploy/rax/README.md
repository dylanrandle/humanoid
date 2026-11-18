# Local RAX runner

With Docker Desktop running and Codex logged in on the Mac:

```sh
uv run python -m humanoid.integrations.rax.runner --build
uv run python -m humanoid.integrations.rax.runner --check-runtime
uv run python -m humanoid.integrations.rax.runner --prepare-only
```

Restart the operator UI after building. Select **Collect toy cars**, then
**Start autonomy**. **Stop** cancels the policy, requests idle, and removes the
runner container. Runs and generated programs are saved under `logs/rax/`.
The launcher mounts the existing Codex login read-only; it does not copy it into
run artifacts or the agent workspace. Model choice and episode settings live in
`src/humanoid/config/rax.py`. This integration is experimental and has not yet
produced a verified cleanup program. Hardware autonomy remains disabled.
New trials allow six exploration episodes of up to 30 minutes each, including
the qualifying rehearsal, followed by three held-out evaluation layouts.

The runtime check exercises Python MCP calls and submitted-program execution in
their actual sandboxes, using dummy responses. It checks private-file and network
isolation without accessing your login, calling a model, or moving the robot.

The image uses Python 3.12, the official Codex CLI 0.169.0 and RAX revision
`4b577aa2ba0039ad01707ede236be9913274180e`. Robot simulation and policies stay in
the normal host stack; Docker runs only RAX and the generated program.
`HUMANOID_RAX_COMMAND` optionally overrides the launcher. For a UI on a different
port, set `HUMANOID_RAX_ENDPOINT` to its Docker-reachable RPC URL.

To resume an interrupted exploration, append `--resume /absolute/path/to/logs/rax/<run>`
to the runner command in `HUMANOID_RAX_COMMAND` (a JSON array of arguments), then
restart the UI and select the same task. **Start autonomy** resumes the saved
RAX conversation, program files, and notes. An interrupted simulation episode
is recovered by RAX; the robot's physical simulation state is not restored.

For another learning trial after one finishes, use `--adapt-from /absolute/path/to/logs/rax/<run>`
instead of `--resume`. Native RAX transfers the saved memory, eligible code, and
compacted conversation into a new trial with new exploration layouts. The source
is mounted read-only inside the private operator directory. Its results and
episode budget stay unchanged; source frames and formal evaluation data are not
made available to the new agent. Store reusable learned code in the RAX workspace's
`code/` directory so it is included in native transfers.

## Test a saved controller

After a learning run finishes, start the UI with its saved controller:

```sh
HUMANOID_RAX_COMMAND='["uv","run","python","-m","humanoid.integrations.rax.runner","--replay-from","/absolute/path/to/logs/rax/<run>"]' uv run start
```

Select **Triskel**, **Simulation**, **Collect toy cars**, and **Start autonomy**
to replay its submitted controller in a fresh simulation.
If there is no submission, a run with one registered bundle can still be tested;
use `--bundle <id>` to choose among several. Add `--seed <number>` for another layout.

Replay runs the frozen program in the same sandbox and realtime stack, without
calling a model or mounting your Codex login. It writes separate artifacts under
`logs/rax/humanoid-replay-*`; the learning run and its evaluation stay unchanged.
A completed replay is not necessarily a successful cleanup: inspect its recorded
`task_complete` result. Hardware autonomy remains disabled.

## Container sandbox settings

The container runs as the host user with all capabilities dropped and
`no-new-privileges`. RAX's own Codex and bubblewrap sandboxes remain enabled.
The container has no Docker socket or device mounts.

`seccomp.json` derives from the [Moby default profile](https://github.com/moby/profiles/blob/main/seccomp/default.json),
retrieved November 15, 2026 (original SHA-256
`3320527d4568265ac97a077609b1bfd4cfd0d789eeb9239b7e094b1526675406`).
It additionally allows `clone`, `unshare`, `mount`, `umount2` and `pivot_root`
so RAX can construct its nested namespaces. Other default syscall restrictions
remain in place. `systempaths=unconfined` lets bubblewrap mount a private proc
filesystem; the kernel still enforces the non-root user's permissions.
These settings apply only to this runner container, not to Docker globally.
