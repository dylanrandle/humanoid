# Agent control

Agents can observe robot state and RGB-D images, then request bounded motions
through the same backend as the operator console. The operator enables motion
by selecting **API** mode.

## Console tasks

Start a Triskel stack using [Getting started](getting-started.md). In the
**Triskel agent** panel, select **Connect ChatGPT**. The backend uses the dashboard
host's existing Codex login or provides a device code and sign-in link. Choose an
image-capable model from the account catalog.

Enter an instruction and select **Run task**. In **Idle**, the agent can inspect
state and cameras. To allow motion, start **API** under **Control inputs**, then
select **API** mode. In simulation, try asking it to move the base 2 cm forward
and report the measured result.

The panel shows progress, results, and errors. **Stop** interrupts the task and
cancels its in-flight command; stopping a motion task requests Idle. Mode changes
and stack shutdown also stop the task. Refreshing the browser reconnects to the
current task. Sign-in, model calls, and execution all run on the dashboard host.

## Python client quickstart

With Triskel running in **Simulation** and **API** enabled as above, run:

```bash
uv run python - <<'PYTHON'
from humanoid.api.clients.robot import RobotClient
from humanoid.types.motion import ControlOperation

client = RobotClient()
try:
    client.wait_until_ready()
    print("Initial joints", client.state().joint_positions)
    result = client.execute(
        ControlOperation.MOVE_BASE, {"delta": [0.02, 0, 0], "duration": 1.2}
    )
    print("Reached", result["reached"], "error (m)", result["position_error_m"])
finally:
    client.close()
PYTHON
```

Select **Idle** when finished. State and camera reads remain available while the
stack runs, independently of motion policy availability. `wait_until_ready()`
waits for observations; `motion_ready()` checks executor readiness.

Use `client.observe()` for camera captures; [Cameras](hardware/cameras.md#understanding-depth)
explains depth registration and missing samples. See [Networking](networking.md#remote-api-access)
to connect from another machine.

## Motion outcomes and retries

Only one motion can be pending or running at a time. The executor checks control
mode, fresh feedback, and configured limits before admitting a request. A completed
motion may still report `reached=False`; inspect the measured result rather than
assuming the requested duration was enough.

If a submission response is lost, query its motion ID before deciding to retry.
An HTTP retry must retain the same request body, ID, session, and deadline; each
Python `execute()` call creates a new ID. Leaving API interrupts outstanding work.
After a restart, reconcile the robot state and [motion session](api.md#runtime-and-motion-lifecycle)
before moving again. If feedback or a policy goes quiet, the controller can continue
tracking its last targets; an interrupted request is not a hardware emergency stop.

Exact operations, arguments, and responses live in the
[robot client](../src/humanoid/api/clients/robot.py),
[motion types](../src/humanoid/types/motion.py), and
[robot routes](../src/humanoid/api/routes/robot.py).

## Perception

On Triskel, the agent can segment camera images with [NanoSAM](models/nanosam.md)
and fill holes in measured depth with [camera depth models](models/cdm.md).
Segmentation supports point, box, grid, and refinement prompts. Depth completion
produces estimates; measured depth remains available separately.

Perception tools reuse a captured observation. Motion or a new observation refreshes
the current view. Saved masks retain observation and calibration information for
downstream consumers through [MaskStore](../src/humanoid/perception/masks.py).
Tool schemas and coordinate conventions live in the
[harness implementation](../src/humanoid/agent/harness/).

## See also

- [Documentation index](README.md)
- [Agent trials](trials.md)
- [Robot backend](api.md)
- [Agent configuration](../src/humanoid/config/agent.py)
