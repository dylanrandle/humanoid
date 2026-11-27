# Robot API for agents

An agent can read robot state and RGB-D images over HTTP and submit bounded motion requests. The operator enables motion by selecting **API** in the console. `MotionPolicy` runs the motions through the same LCM orchestrator and controllers as the other policies.

## Try it in MuJoCo

1. Run `uv run start`. In the operator console, select **Triskel** and **Simulation**, then start the main stack and wait for the mode to show **Idle**.
2. Local agents can already read state and camera frames in Idle. Under **Control inputs → API**, click **Start** when you want to enable motion.
3. Select **API** to enable motion. From a terminal on the same machine, request a 2 cm translation along the base's local x-axis.

   ```sh
   uv run python - <<'PYTHON'
   from humanoid.api.clients.robot import RobotClient
   from humanoid.types.motion import ControlOperation

   client = RobotClient()
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

4. Select **Idle** to end agent control. Stop **API** to stop the motion executor. The unified API remains available after the policy or main stack stops.

Call `client.observe()` for RGB and synchronized measured depth in the RGB pixel grid.
Read metric depth at an RGB pixel with `view.depth.data[y, x] * view.depth.depth_scale`.
Zero samples are holes. If a fresh synchronized aligned pair is unavailable,
the view contains RGB and `view.depth is None`. Request `client.observe(depth_view="native")`
for depth in the sensor's native optical frame with its own calibration.

An agent on another machine can reach the default loopback listener through an SSH tunnel such as `ssh -L 8765:127.0.0.1:8765 triskel`. The listener binds only to `127.0.0.1`. Any process on that host can access it while it is running. Use `uv run start --port <port>` to change the listener port. Set `HUMANOID_API_URL` for clients, or pass an explicit endpoint. For a different local port, pass `ApiEndpoint("http://127.0.0.1:<port>")` to the client.

## HTTP contract

The API lives in `humanoid.api`; all routes share the dashboard listener. The API remains available with the motion policy stopped. `client.wait_until_ready()` waits for observations; `client.motion_ready()` reports whether the executor has fresh feedback in API. Motion admission is always checked again on submission.

POST bodies use `Content-Type: application/json`.

`runtime_id` identifies the robot runtime and changes when the stack, robot, scene or runtime
is replaced. `motion_session_id` identifies the motion executor and also changes when that
executor restarts. Read-only clients and captures remain valid across motion restarts. A client
that has pinned motion authority must be recreated before controlling a replacement executor.
`RobotClient.runtime_id` and `RobotClient.motion_session_id` expose these distinct identities.

Episode recording requests use `{runtime_id, episode_id}`. They remain active through motion
restarts and finalize on runtime shutdown/reset or an explicit owner stop. Retrying an old
episode stop cannot stop a newer recording.

| Endpoint | Response |
| --- | --- |
| `GET /api/state` | `runtime_id`, `motion_session_id`, operator mode, joint feedback, feedback age in seconds and `motion_ready` from the executor |
| `GET /api/camera` or `GET /api/camera?depth_view=aligned` | RGB and aligned measured depth PNGs, frame IDs, intrinsics, depth scale, timestamps and age. Missing aligned depth is JSON `null`. |
| `GET /api/camera?depth_view=native` | RGB and native measured depth with their respective calibration. |
| `GET /api/camera?depth_view=both` | RGB and native depth with their own calibration, plus RGB-registered depth when available. |
| `GET /api/capture` | Per-camera RGB, native and registered depth, nearest joint state, and current operator mode. |
| `POST /api/release_autonomy` | Requests release for the current `motion_session_id`. The robot enters Idle only if API is still active. |
| `POST /api/motions` | HTTP 202 with the motion ID and current status |
| `GET /api/motions/<id>?motion_session_id=…` | Motion status and, on completion, measured result |
| `POST /api/motions/<id>/cancel` | Updated status for that motion. Send `{"motion_session_id": "…"}` as the body. |

Use `motion_session_id` from `/api/state` and a unique client-generated motion ID.

```json
{
  "motion_session_id": "session-from-state",
  "motion_id": "move-001",
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
| `hold` | Interrupt the active motion, hold the measured tool pose, stop the base and retain the gripper target. |

All operations accept `duration` in seconds, default 1 and maximum 2. One motion can be pending or running at a time. Overlapping motion gets HTTP 409. Other 409 responses include stale feedback, missing fresh camera frames and an inactive control mode. Cross-origin browser requests return 403 and malformed requests return 400. The policy also checks workspace, displacement and joint limits against feedback on the execution tick.

Poll the motion ID until its state leaves `queued` or `running`. Terminal states are `completed`, `rejected`, `cancelled`, `expired` and `failed`. A completed motion reports `reached` and the measured position, orientation and gripper errors. The duration can elapse before the robot reaches its target. Leaving API interrupts outstanding motions.

During a motion the motion policy uses measured odometry to calculate the remaining
base displacement. It applies a speed-bounded correction to the last routed base
pose in `ROBOT/JOINT_COMMAND`. If measured feedback exceeds `feedback_timeout`,
the policy interrupts the motion and stops issuing targets. The robot controller
continues tracking its last targets if the policy or feedback stream goes quiet.

On a mode handoff the controller holds the position-controlled joints at their last routed
`ROBOT/JOINT_COMMAND` targets, including targets sent during homing. It uses the
measured joints when it has no usable command. The base starts at measured
odometry with zero commanded velocity. Operational-space policies use the same
arm and gripper reference for their first action.

If a submission response is lost, resend the **same JSON body**, including the session ID, motion ID and deadline. The executor returns the existing status and measured result for that ID. A retry neither repeats the movement nor extends its start deadline. Reusing the ID for different arguments fails. After a backend, stack, configuration or policy restart, read the new state and reconcile the robot's position before deciding on another movement. Each call to the Python client's `execute()` creates a new motion ID. Use the HTTP contract directly when the caller needs to retain its own IDs.

`MotionService` sends typed messages over a private multiprocessing pipe to the managed `MotionPolicy` node. Background workers handle pipe I/O; the control tick never waits for HTTP, model calls, or pipe writes. A 202 response means the executor acknowledged the motion; it does not mean the robot reached its target. If an acknowledgement times out, query the same ID before retrying. A pipe disconnect cancels outstanding work, emits the established hold/zero-base action and requests Idle through the orchestrator. Heartbeat loss uses `MOTION_PROTOCOL_CONFIG.contact_timeout_seconds`; executor restarts invalidate the session and never replay earlier requests.

Future MCP tools should call the same backend services; no MCP server is included yet.

## Triskel agent harness

`AgentTask` describes the requested work (`instruction`). `AgentExecution` binds that task to
an execution model, robot, operator mode, runtime identity and optional motion session identity.
The dashboard admits an execution and sends it to the background worker, which validates its
identities before running. CLI exploration trials reuse `AgentTask` directly, with their model
thread, robot connection, mode and time budget supplied by the trial/episode lifecycle; they
do not construct placeholder session IDs.


Open the operator console, select **Triskel**, and start the main stack. Start **API** for motion tasks. In the **Triskel agent** panel, select **Connect ChatGPT**. The
console uses the dashboard host's existing Codex ChatGPT login or shows a
device code and sign-in link. Select an image-capable model from the account
catalog. The panel selects Astra 6.2 when available.

Enter an instruction and select **Run task**. In **Idle**, the agent answers
using measured robot state and one labeled RGB image per camera. It can request
depth measurements from the same capture. In **API**, it can
also issue bounded tool, base, gripper and hold commands. For a simulation
test, ask it to move the base 2 cm forward and report the measured result.
Each motion refreshes the observation. Perception tools can work on the same
capture across turns. Each console task has a 10-minute wall-clock budget for
model decisions, perception, and motion. The panel displays progress, the final
answer, and any errors.

**Stop** interrupts the worker and cancels its in-flight command. Stopping a
running motion task requests Idle. Changing control modes in the console or
stopping the motion policy or stack also stops the task. Refreshing the browser
reconnects to the current task. Device sign-in, model calls, and task execution
run on the dashboard host, including on the Jetson.

### Instance segmentation

On Triskel the agent can segment the observed D435 or D405 RGB image in Idle
or API. The first segmentation call encodes RGB with NanoSAM, and later
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

### Read or complete depth

`read_depth(observation_id, view, points_xy)` returns axial depth in metres, or `null` for
missing samples. Use `registered` for RGB pixels and `raw` for native depth pixels.
Valid simulated depth is ground truth. For holes in hardware depth, `complete_depth`
uses CDM to estimate and cache an RGB-aligned map. It returns the valid pixel count
and depths at the requested RGB pixels. [CDM installation](../../../models/cdm/README.md)

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

## Toy cleanup trials

Sign the harness in to ChatGPT, start Docker, and run the trial from the repository root.

```sh
uv run python -m humanoid.agent.harness.trial --explore 3 --test 3 --record toy_cleanup --items 1
```

The agent explores three seeded MuJoCo layouts with one cube, then writes a Python controller.
The runner tests that controller in Docker on the next three seeds. A toy counts as deposited
when it settles inside the bin after release, supported by the bin floor or another deposited toy.

Place shared trial options before the task name. Use `--seed` to choose the first layout and
`--explore-seconds` or `--test-seconds` to set the per-episode time budgets.
Use `toy_cleanup --items 12` for the full toy set.
To evaluate a saved controller directly, pass `--replay PATH/TO/controller.py --seed N`,
where `N` is the first test layout.

During exploration, the agent sees RGB and requests depth by observation ID. During evaluation,
`step(obs, memory)` receives measured robot state and per-camera `rgb`, `registered_depth_m`, and
`raw_depth_m` arrays sampled every eighth pixel. Use the RGB calibration for registered depth
and `obs["cameras"][name]["raw_depth"]` for native depth.

**Hardware.** Start the real stack and motion policy, then add `--backend real --robot triskel`.
For each episode, set up the layout, home the robot, and select **API**. The runner
requests **Idle** before asking you to grade the result. For remote control and recording, forward the unified API port. Set `--endpoint` to the forwarded address.

### Review results and recordings

The runner prints the run directory under `logs/trials/`. Use `--output` to change it.

| File in the run directory | What to inspect |
| --- | --- |
| `explore-<seed>/index.html` | Each decision's intent, camera images, tool arguments, and result. Reload during a run or copy the whole episode folder to view it elsewhere. |
| `controller.py` | The generated controller used for evaluation and replay. |
| `test-<seed>.json` | Motion commands, measured outcomes, score, and the reason the episode ended (`ended_by`). |

`--record` saves one LCM log per episode, including camera and robot channels. The episode
JSON gives its path in `lcm_recording`. Simulation logs go under the run's `recordings/`
directory. Hardware logs stay on the robot. Every exploration gets an HTML review.

With FFmpeg installed, this command exports synchronized RGB and colorized depth MP4s from a log or bundle.

```sh
uv run python scripts/export_camera_videos.py /path/to/recording.lcm
```

### Add a simulation task

Implement `SimulationTask` with a scene, goal, and `score(data)` method. Override
`configure_scene(spec, seed)` for randomized layouts. Register the task's CLI argument
function in `TASKS`. [Toy cleanup example](../simulation/tasks/toy_cleanup.py) · [Task registry](../simulation/tasks/__init__.py)


## Shared contracts

`types/agent.py` owns tasks, executions, decisions, and decision-loop context.
`types/trial.py` owns submitted programs, program memory, evaluation traces, and scoring.
Both use tool-call wrappers and the `{operation, error}` failure shape from `types/tools.py`.
Motion argument schemas live in `types/motion.py` and validate API, agent, and trial inputs.
The API accepts omitted `duration`; model response schemas require every field explicitly.
Model-specific schema formatting lives in `agent/harness/model.py`.

`ObservationSummary` in `types/observation.py` describes the measured state presented to
agents and trial programs. `CameraSummary` and its calibration types live in `types/camera.py`.
Use `obs["cameras"][obs["primary_camera"]]` for the primary camera's observation ID,
frame number, depth availability, and intrinsics (including image width and height).
These summaries are JSON-friendly views of native `Observation` and camera data.

`MotionTarget` describes a bounded movement over a duration. `Action` remains the separate
per-tick policy output used by motion and other policies.
