# Remote robot control

`RemoteControlPolicy` runs on the normal policy/environment/LCM/controller path.
The HTTP listener runs in its own thread in the managed control process. It has
no Flask, UI, or RAX dependency. Network requests never publish motor commands.

Start a simulation or hardware stack, then run the service independently:

```sh
# Set HUMANOID_CONTROL_TOKEN to a private operator-chosen credential first.
uv run python -m humanoid.remote_control
```

The default address is `127.0.0.1:8767`. Use an SSH tunnel to reach a service on
another machine. `--host` and `--port` are available for an explicitly configured
network. HTTP does not provide encryption. Activation stays with the operator:
the existing orchestrator must be in `AUTONOMY` mode before motion executes.
The managed autonomy process starts the same service with a fresh credential.

Every request needs `Authorization: Bearer <token>`.

| Endpoint | Behavior |
| --- | --- |
| `GET /v1/state` | Current session ID, mode, joint feedback, and feedback age |
| `GET /v1/camera` | Native RGB/depth PNGs in base64, separate intrinsics, frame IDs, depth scale, timestamps, and age |
| `POST /v1/commands` | Submit a command; return immediately with its ID and status |
| `GET /v1/commands/<id>` | Read status and measured result |
| `POST /v1/commands/<id>/cancel` | Interrupt that command; body `{}` |

Submit with the session ID returned by `/v1/state`:

```json
{
  "session_id": "session-from-state",
  "command_id": "unique-client-command-id",
  "operation": "move_base",
  "arguments": {"delta": [0.01, 0, 0], "duration": 0.5},
  "expires_in_seconds": 1.0
}
```

Operations are `move_tool` (`position`, unit `quaternion` in x/y/z/w order),
`move_base` (`delta` in x/y/yaw), `set_gripper` (`positions` in joint radians),
and `hold`. All accept `duration`, defaulting to one second. Tool poses use the
robot's configured tool-command frame; base deltas use its pose at execution
start. The policy validates motion against its config and the robot's limits.

One command can be pending or running. Overlapping motion receives HTTP 409;
`hold` can interrupt it. Explicit cancel/hold retains the existing gripper squeeze.
Mode changes, stale feedback, and shutdown reset held references. Completed
commands stop base motion and continue the arm/gripper hold. Completion reports
measured errors and `reached`; it does not establish grasp success.

Statuses are `queued`, `running`, `completed`, `rejected`, `cancelled`, `expired`,
and `failed`. Validation against robot feedback occurs on the next policy tick.
Start expiry uses the receiving host's clock. Server monotonic timestamps are
metadata, not directly comparable with client clocks; use the returned ages.

Retrying an identical ID returns its existing status, without restarting motion
or refreshing its deadline. Reusing it for different arguments is rejected. IDs
are retained for the entire session; when history fills, new commands are refused.
A service restart changes the session ID so old submissions cannot be replayed.

Python callers can use `RemoteControlClient` from `humanoid.remote_control.client`
with a `RemoteControlEndpoint`. `execute()` waits on the calling thread while the
policy runs independently. RAX is one adapter of this general interface. Task
setup, learning runs, and evaluation remain in the RAX/operator integration.

The Jetson operator console also serves `GET /api/hardware/observation` for
read-only clients attached to a running real Triskel stack. This route works
in Idle mode. It returns measured state, tool pose, gripper limits and a paired
native RGB-D snapshot with separate intrinsics, optical-to-base transforms,
frame numbers, timestamps, depth scale and lossless PNGs. Use the console SSH
tunnel. A stopped stack, replay, simulation, or missing fresh data returns HTTP
409. The [read-only RAX trial](../../../deploy/rax/README.md#read-only-trial-on-the-real-triskel)
uses this same endpoint.
