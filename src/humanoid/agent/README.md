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
| `GET /v1/camera?depth_view=both` | RGB and native depth with their own calibration, plus RGB-registered depth when available. |
| `GET /v1/capture` | All three views from each named camera, the joint state nearest each capture time, and the current operator mode. |
| `POST /v1/idle` | Releases Autonomy for the given `session_id`. The orchestrator switches to Idle only if Autonomy is still active. |
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
using measured robot state and one labeled RGB image per camera. It can request
depth measurements from the same capture. In **Autonomy**, it can
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

Simulated depth gives ground-truth distances at valid pixels, so skip CDM in simulation.
Use `read_depth(observation_id, view, points_xy)` for measured axial depth in metres.
Choose `registered` for RGB pixel coordinates or `raw` for coordinates on the native
depth sensor. A zero measurement returns `null`.
On hardware, call `complete_depth(observation_id)` to run CDM when missing RealSense
depth affects the task.
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

### Run a simulated cleanup trial

With Docker running and the harness signed in to ChatGPT, run from the repository root:

```sh
uv run python -m humanoid.agent.harness.trial toy_cleanup --record
```

Each simulated episode starts an isolated digital twin, controller, orchestrator and Agent API.
Physics and cameras advance while the agent deliberates and while its generated controller
runs. With `--record`, each simulated episode records its private
LCM bus, including simulation state, robot commands and feedback, and both cameras'
RGB and depth channels. Its bundle contains `robot.json` and `recording.lcm` under
the run's `recordings/` directory. The episode JSON gives the path in `lcm_recording`.
Each episode starts with fresh physics, controllers and camera state. The seed sets the room dimensions, bin and toy positions, and
the colors of the toys, bin, walls and floor. The agent then writes a Python
controller. The runner freezes it and tests it on the next three seeds. Each
decision runs in Docker with measured RGB-D, robot state and the program's own
memory. A deposit counts when the toy's center is inside the bin, the robot has
released it, and the bin floor supports it directly or through other deposited toys.
The evaluator checks again after a settling interval.

Controller observations express `tool_pose` in `tool_pose_reference_frame`
(the base frame on Triskel, world for a fixed-base robot). `gripper_positions`
and `gripper_limits` follow configured joint order, with each joint's unit
listed in `gripper_units`. On Triskel, `obs["images"]["base"]` and
`obs["images"]["wrist"]` each contain `rgb`, `registered_depth_m` and `raw_depth_m` sampled every
eighth pixel. Each corresponding entry in `obs["cameras"]` supplies native intrinsics,
optical-frame pose in the tool command reference frame, frame number and device timestamp.
Registered depth uses RGB calibration. Raw depth uses the intrinsics and pose in
`obs["cameras"][name]["raw_depth"]`. Exploration receives an RGB image per camera and
can select either camera's observation ID for depth measurements, segmentation or completion.

To run the same trial on hardware, start the real stack and Agent API, then use
`--backend real --robot triskel` with the local or forwarded `--endpoint`. The operator
arranges each layout, homes the robot, enables Autonomy and grades the outcome after the
runner requests Idle. Both configured Triskel cameras must supply registered depth before
a trial decision. Add `--record` to start the stack's LCM logger
automatically before the first decision of each physical episode and stop it when
the runner releases Autonomy, before operator grading. It captures the LCM channels,
including RGB and depth from both cameras. The episode JSON's `lcm_recording` field
gives the log path on the robot, in its standard recording bundle with `robot.json`.
For a remote run, forward the operator console port as well as the Agent API port.
Use `--console-url` if the forwarded console is at a different address.

The console and hardware trials both POST to `/api/logging/start` and
`/api/logging/stop`. The Record button sends `{}` and receives the console status.
A trial sends `episode_id` and `session_id` on both requests and receives the
acknowledged logger status, including the recording path.

To export camera videos from a simulation or hardware log, install FFmpeg and run
the following from the repository root.

```sh
uv run python scripts/export_camera_videos.py /path/to/recording.lcm
```

The script also accepts a recording bundle directory. It writes one MP4 per recorded
RGB, depth and registered-depth stream to `<log-stem>-videos/` beside the log.
Depth uses the console's distance colors. All videos share the recording timeline.
Use `--output` to choose the destination and `--fps` to set the output frame rate.

By default each room contains one cube and a bin. Use `--items 12` to run the full toy set or replay an earlier twelve-toy trial. Each exploration episode gets 10 minutes of wall-clock time. The generated controller
gets 1 minute per evaluation episode, including replay. Set `--explore-seconds` and
`--test-seconds` to override these limits. Timing starts after episode setup and
includes observations, model or controller decisions, and motions. A synchronous
perception call can finish after expiry. Reset, final settling, and saving
artifacts are outside the decision budget.
Select a robot with `--robot` (defaults to `HUMANOID_ROBOT`, normally Triskel).
Vision trials require a configured RGB-D camera. Triskel supplies both base and wrist views.
Use `--seed`, `--explore` and `--test` to change the episode split.
The episode JSON records why execution ended in `ended_by`.
The run directory under `logs/trials/`
contains the settings, exploration notes, generated `controller.py`, and test
results with each command, measured outcome and deposited toy name. The first
evaluation downloads the uv Python 3.12 container image if needed.

### Inspect exploration step by step

Open `explore-<seed>/index.html` inside the run directory. Use the step selector
or the left and right arrow keys to follow the agent's stated intent, its command
and arguments, and the returned result. Each step shows the labeled images sent
to the model, including pinned views and segmentation overlays. Click an image
for its full resolution, or expand the model input to see the instruction,
remaining budget, robot pose, camera calibration and preceding feedback.

Each page updates on disk as the episode runs. Reload it to see new decisions.
For a remote trial, copy the episode's entire `explore-<seed>/` directory,
including `images/`, and open the page locally. These reviews are saved for
every exploration episode, independently of `--record`.

To repeat an evaluation without a model call, pass `--replay PATH/TO/controller.py`
and set `--seed` to the first layout to test.

### Add a simulation task

Implement `SimulationTask` from `humanoid.types.trial` in
`src/humanoid/simulation/tasks/`. Select a `SimulationScene`, provide the goal
used in exploration and controller generation, and implement `score(data)` to
return success and JSON metrics. Multiple tasks can use the same scene. For a
new scene, add a `SimulationScene` member and a matching
`src/humanoid/simulation/scenes/<value>.xml` file.

Override `configure_scene(spec, seed)` for per-episode setup or randomization.
Register a CLI argument
function in `humanoid.simulation.tasks.TASKS`. It can add task-specific flags
and returns a factory taking the parsed arguments. The shared runner handles reset, observations, motion,
recording, exploration, generation and held-out evaluation.
