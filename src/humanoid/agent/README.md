# Agent control API

An agent can read robot state and RGB-D images over HTTP and submit bounded motion commands. The operator enables motion by selecting **Autonomy** in the console. `AgentPolicy` runs the commands through the same LCM orchestrator and controllers as the other policies.

## Try it in MuJoCo

1. Run `uv run start`. In the operator console, select **Triskel** and **Simulation**, then start the main stack and wait for the mode to show **Idle**.
2. Under **Control inputs → Agent API**, click **Start**. Local agents can now read state and camera frames while the robot is in Idle.
3. Select **Autonomy** to enable motion. From a terminal on the same machine, request a 2 cm translation along the base's local x-axis.

   ```sh
   uv run python - <<'PYTHON'
   from humanoid.agent.client import AgentControlClient
   from humanoid.types.agent import ControlOperation

   client = AgentControlClient()
   try:
       client.wait_until_ready()
       print("Initial joint positions", client.state().joint_positions)
       result = client.execute(
           ControlOperation.MOVE_BASE, {"delta": [0.02, 0, 0], "duration": 1.2}
       )
       print("Reached", result["reached"], "position error (m)", result["position_error_m"])
       print("Final joint positions", client.state().joint_positions)
   finally:
       client.close()
   PYTHON
   ```

4. Select **Idle** to end agent control. Click **Stop** in the Agent API row to shut down the listener. Stopping the main stack also stops the API.

Call `client.observe()` for RGB and synchronized measured depth in the RGB pixel grid.
Read metric depth at an RGB pixel with `view.depth.data[y, x] * view.depth.depth_scale`.
Zero samples are holes. If a fresh synchronized aligned pair is unavailable,
the view contains RGB and `view.depth is None`. Request `client.observe(depth_view="native")`
for depth in the sensor's native optical frame with its own calibration.

An agent on another machine can reach the default loopback listener through an SSH tunnel such as `ssh -L 8767:127.0.0.1:8767 triskel`. The listener binds only to `127.0.0.1`. Any process on that host can access it while it is running. Set `HUMANOID_CONTROL_PORT` in the dashboard environment to change the listener port. For a different local port, pass `AgentControlEndpoint("http://127.0.0.1:<port>")` to the client.

## HTTP contract

POST bodies use `Content-Type: application/json`.

| Endpoint | Response |
| --- | --- |
| `GET /v1/state` | Current session ID, operator mode, joint feedback and feedback age in seconds |
| `GET /v1/camera` or `GET /v1/camera?depth_view=aligned` | RGB and aligned measured depth PNGs, frame IDs, intrinsics, depth scale, timestamps and age. Missing aligned depth is JSON `null`. |
| `GET /v1/camera?depth_view=native` | RGB and native measured depth with their respective calibration. |
| `GET /v1/capture` | Named RGB-D views. Each view includes the joint state nearest its host capture time and the current operator mode. |
| `POST /v1/commands` | HTTP 202 with the command ID and current status |
| `GET /v1/commands/<id>` | Command status and, on completion, measured result |
| `POST /v1/commands/<id>/cancel` | Updated status for that command. Send `{}` as the body. |

Use the session ID from `/v1/state` and a unique client-generated command ID.

```json
{
  "session_id": "session-from-state",
  "command_id": "move-001",
  "operation": "move_base",
  "arguments": {"delta": [0.02, 0, 0], "duration": 1.2},
  "expires_in_seconds": 1.0
}
```

| Operation | Arguments |
| --- | --- |
| `move_tool` | Absolute `position` in metres and unit `quaternion` in x, y, z, w order. Tool targets use the configured base frame for a mobile robot and the world frame for a fixed-base arm. |
| `move_base` | `delta` as forward x (m), lateral y (m), and yaw (rad), relative to the mobile base at command start. |
| `set_gripper` | `positions` in configured gripper-joint order, using each joint's native units (rad for Triskel). |
| `hold` | Interrupt the active motion and hold the measured pose while retaining the gripper target. |

All operations accept `duration` in seconds, default 1 and maximum 2. One command can be pending or running at a time. Overlapping motion gets HTTP 409. Other 409 responses include stale feedback, missing fresh camera frames and an inactive control mode. Cross-origin browser requests return 403 and malformed commands return 400. The policy also checks workspace, displacement and joint limits against feedback on the execution tick.

Poll the command ID until its state leaves `queued` or `running`. Terminal states are `completed`, `rejected`, `cancelled`, `expired` and `failed`. A completed command reports `reached` and the measured position, orientation and gripper errors. The duration can elapse before the robot reaches its target. Leaving Autonomy interrupts outstanding commands.

If a submission response is lost, resend the **same JSON body**, including the session ID, command ID and deadline. The service returns the existing status and measured result for that ID. A retry neither repeats the movement nor extends its start deadline. Reusing the ID for different arguments fails. After a service restart, read the new state and reconcile the robot's position before deciding on another movement. Each call to the Python client's `execute()` creates a new command ID. Use the HTTP contract directly when the caller needs to retain its own IDs.

## Triskel agent harness

Open the operator console, select **Triskel**, and start the main stack and
**Agent API**. In the **Triskel agent** panel, select **Connect ChatGPT**. The
console uses the dashboard host's existing Codex ChatGPT login or shows a
device code and sign-in link. Select an image-capable model from the account
catalog. The panel selects Astra 6.2 when available.

Enter an instruction and select **Run task**. In **Idle**, the agent answers
using measured robot state and a labeled RGB image from each camera. In **Autonomy**, it can
also issue bounded tool, base, gripper and hold commands. For a simulation
test, ask it to move the base 2 cm forward and report the measured result.
Each motion refreshes the observation. Perception tools can work on the same
capture across turns. Each console task has a 10-minute wall-clock budget for
model decisions, perception, and motion. The panel displays progress, the final
answer, and any errors.

**Stop** interrupts the worker and cancels its in-flight command. Stopping a
running motion task requests Idle. Changing control modes in the console or
stopping the Agent API or stack also stops the task. Refreshing the browser
reconnects to the current task. Device sign-in, model calls, and task execution
run on the dashboard host, including on the Jetson.

### Instance segmentation

On Triskel the agent can segment the observed D435 or D405 RGB image in Idle
or Autonomy. The first segmentation call encodes RGB with NanoSAM, and later
prompts reuse the encoding. [NanoSAM installation](../../../models/nanosam/README.md)

| Tool | Agent inputs | Result |
| --- | --- | --- |
| `segment_everything` | `observation_id`, optional `region_xyxy` | Numbered proposals from a grid of prompts. A region restricts seed locations and masks can extend outside it. |
| `segment_at_point` | `observation_id`, `point_xy`, `return_alternatives` | An object at the pixel, or distinct ranked interpretations of that point. |
| `segment_in_box` | `observation_id`, `box_xyxy`, `return_alternatives` | An object outlined using its approximate bounding box. |
| `refine_mask` | `mask_id`, `include_points`, `exclude_points` | A new mask on the same observation, using the previous logits and prompt history. Supply at least one correction. |

Coordinates are integer pixels in the original RGB image, measured from the
top left. Boxes use inclusive left, top, right and bottom coordinates. Each segmentation call
returns an RGB overlay and a numbered list with mask IDs, bounding boxes, pixel
counts and the decoder's predicted IoU score. Higher scores rank first.
RGB segmentation also works on an RGB-only view.

### Depth completion

Call `complete_depth(observation_id)` when a task needs a dense metric depth map.
It uses synchronized RGB and aligned measured depth from the pinned observation.
It saves the full RGB-aligned float32 map under the observation ID and returns the valid
pixel count. Supply native RGB pixels in `points_xy` to read their axial depths in
metres. The `depths_m` values follow the requested order, with `null` for invalid
pixels. Repeated calls for the same observation read the cached map. [Depth model installation](../../../models/cdm/README.md)

### Using a saved observation

Motion or `observe_again` pins a new view. Completion and refinement require the current
view. Downstream readers can resolve a saved selection after a view change or
worker restart.

Pass the opaque `mask_id` to a downstream tool running against the same
artifact store (by default `$HUMANOID_DATA_DIR/logs/segmentation`). Each
saved observation stores its original RGB, camera serial, frame numbers, capture
timestamps and RGB intrinsics. Before completion, `completed_depth_m` is `None`.
After completion, resolving any mask from that observation also loads its full
float32 axial depth map in metres. For a depth-only workflow, use
`MaskStore().resolve_depth(observation_id)` without creating a mask.

```python
import numpy as np
from humanoid.perception.masks import MaskStore

target = MaskStore().resolve(mask_id)
if target.completed_depth_m is None:
    raise ValueError(f"Request complete_depth for observation {target.observation_id}.")
valid = target.mask & np.isfinite(target.completed_depth_m) & (target.completed_depth_m > 0)
v, u = np.nonzero(valid)
depth_m = target.completed_depth_m[v, u]  # Use (u, v) and target.intrinsics for deprojection.
```
