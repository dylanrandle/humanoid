# Native MuJoCo simulation

Simulation replaces the hardware driver while retaining the normal orchestrator,
robot controller, logger, visualizer, and homing policy.

## Run collection

Simulation starts with **Empty scene**, the robot-only environment for teleoperation
and manual recording. It works with every robot. Select it again while Idle to clear
a task scene without restarting the stack. Automated collection is disabled for it.

Start `uv run start`, select **Simulation** and the task's robot, and start the stack.
In **Simulation task**, choose the task, successful episode count, attempt budget,
starting seed, simulation time limit, and recording directory.

Task and seed changes load a preview when the stack is Idle and manual recording is
stopped. **Start collection** runs the expert; **Stop collection** flushes the current
attempt and returns to Idle. The stack and viewer remain running. Every attempt uses
the selected task's `create_engine(seed)` to build a fresh world. A bare simulator
contains only the robot.

Configure and run collection through the operator UI. Seeds increment on every attempt,
including failures. Only validated successful episodes count; exhausting the attempt
budget marks the run as failed.

## Implementation boundaries

| Module | Responsibility |
| --- | --- |
| `model.py` | Import the robot URDF into an `MjSpec`; add actuators, mimic constraints, and a planar mobile root |
| `binding.py` | Resolve MuJoCo and Pinocchio joints by name, independently of array ordering |
| `engine.py` | Compile the robot and task scene; reset, advance physics, and export robot/scene state |
| `nodes/robot/simulation.py` | Consume routed joint commands, enforce the watchdog, and own physics advancement |
| `client.py` | Request scene resets, observe snapshots, and publish expert targets |
| `nodes/collection.py` | Handle scene previews and collection batches in an independent process |
| `collection/collector.py` | Record one episode with timestamped observations and applied commands |
| `collection/log.py` | Audit the finished raw LCM recording |

The stack launches the same independent controller, orchestrator, logger, collection,
visualizer, and homing processes for both runtimes. Only the driver changes.
`COLLECTION/REQUEST` carries prepare/start/stop requests; `COLLECTION/STATUS` reports
progress. The collection node does not start or own the stack.

The simulator handles `SIMULATION/RESET` with a registered task and seed, and returns
`SIMULATION/RESET_STATUS`. Requests are retried with the same ID and acknowledged
without resetting twice. Resets require an Idle orchestrator and stopped logger;
stale commands are discarded. The existing controller follows the new robot state
through its ordinary Idle behavior.

`SIMULATION/STATE` supplies timestamped robot and world snapshots. The collection
client loads the same task model for kinematics, geometry queries, and rendering,
then updates it from these snapshots. It never advances local physics. Rendering and
expert computation do not pause the driver or other nodes.

Experts publish `SYSTEM/TOOL_COMMAND` and `SYSTEM/BASE_COMMAND`. The orchestrator
routes these through `RobotControllerNode` to `ROBOT/JOINT_COMMAND`. Joint-space
expert waypoints become tool references, and base velocities become absolute base
targets. `LOGGING/REQUEST` starts/stops recording, optionally choosing an output
directory for that recording. Manual recording keeps its normal defaults.

## Model and timing

The model builder preserves URDF inertias and visual meshes. Multi-part OBJ faces
are merged for MuJoCo import; Triskel's concave fingers use decomposed contact shapes
from `robots/triskel.py`. Robot geoms collide with scene geoms. Self-collision remains
disabled until SRDF exclusions are translated into MuJoCo collision exclusions.

Triskel's gripper torque, friction, contact solver, and coupling solver settings
belong to `RobotConfig.gripper.simulation`. Its `GripperSimulationConfig` shape lives
in `types/robot.py`; the provisional profile lives in
[config/robot/triskel.py](../config/robot/triskel.py). These values remain uncalibrated
pending system identification. `motor_torque_limit=None` preserves the URDF effort
limit. Setting `RobotGripperConfig.simulation=None` also uses native contact and
constraint defaults. Neither changes the collision shapes or URDF mimic ratios.
Cleanup records this profile under `task_config.robot.gripper.simulation`.
`MujocoSimulationConfig` contains only settings shared across robot models.

Mobile roots use x/y slide joints and a yaw hinge. Commands and reported velocities
are transformed between world and body axes. Wheel joints are also simulated.
Actuator targets are applied directly to MuJoCo's position and velocity servos.
Reset initializes targets at home; the watchdog clears velocity targets while
position-controlled joints continue holding. Matching actuator dynamics to real
hardware is deferred to system identification.

The OSC and base controller integrate their commands open-loop in every active mode,
for both simulation and hardware. Measured state initializes the controllers and
resynchronizes them while inactive or before the first command. Active solves use
the integrated commanded configuration, including the commanded gripper aperture.
The original IK limit checks remain in place. Tracking differences can accumulate;
measured state remains available for visualization and expert task checks, but does
not continuously replace the controller's internal configuration. Actuator system
identification remains subsequent work.

The generic physics timestep is 5 ms with `implicitfast`; Cleanup uses Euler at
0.25 ms. Controlled lifts exposed significant slip with implicit actuator damping,
while Euler and RK4 agreed on submillimeter displacement. Cleanup retains elliptic
friction, high friction impedance, and NoSlip post-processing. The driver retains its
normal 200 Hz loop across scene changes, accumulating fractional physics substeps
when necessary. The controller and orchestrator retain their normal rates. The expert
publishes targets at up to 50 Hz and samples at up to 10 Hz, depending on computation
and rendering time. The standard 250 ms watchdog stops stale velocity commands.

## Add a task

Implement `SimulationTask` in `tasks/<task>/` and register it in `tasks/registry.py`:

- The constructor receives the running stack's `robot_config`. Set `robot` on the
  task class to restrict it to one robot, or leave it unset to support every robot.
  Scene-only tasks set `collectable = False`.
- `create_engine(seed)` owns scene construction and seeded randomization. Pass a
  `scene_builder` to `NativeMujocoEngine` to compose the world before compilation.
- `run(runtime)` supplies expert motion references through the shared stack.
- `success(engine)` independently validates completion.
- `cameras()`, `objects(engine)`, `state(runtime)`, and `metadata()` describe observations,
  privileged labels, and reproducible settings.
- `observation_errors(runtime)` optionally validates each RGB sample.

Keep configuration instances in `config/tasks/`, shared types in `types/tasks/`, and
tests in `tests/simulation/tasks/`. See [Cleanup](tasks/cleanup/README.md) for its
physical grasping, navigation, cameras, and diagnostic visibility measurements.

## Recording contract

Schema version 4 uses the existing recording bundle:

```text
<recording-id>/
  robot.json       robot configuration
  recording.lcm    raw messages written by lcm-logger
  episode.json     task configuration, provenance, outcome, and integrity audit
```

| Channel | Default rate | Contents |
| --- | --- | --- |
| `ROBOT/STATE` | 200 Hz | Driver state in Pinocchio ordering |
| `ROBOT/JOINT_COMMAND` | Controller rate | Routed position and body/wheel velocity targets |
| `SIMULATION/STATE` | 50 Hz | World snapshot and last applied command with its own timestamp |
| `SIMULATION/SAMPLE` | Up to 10 Hz | Snapshot state, RGB, object states, labels, and last applied command |
| `SIMULATION/SCENE` | 20 Hz | Viewer geometry |
| `COLLECTION/EPISODE` | Boundaries | Task identity, runtime, clock, outcome, counts, and metadata |

Observation and command timestamps use `host_monotonic`, matching the ordinary robot
streams. Wire timestamps are nanoseconds; LCM receipt times must not replace sensor
timestamps. Images and labels describe the received simulator snapshot, and the
last applied command keeps its application timestamp. `action_index` is -1 because
there is no synchronized control trajectory. Episode action counts count policy
targets, not controller ticks. Simulation time still determines the episode time limit.

Samples embed shared camera, object-state, and task-annotation messages. Camera world
poses, object states, and task labels are privileged supervision. Cleanup's intended
policy observes the four current RGB views (wrist and three base cameras). See the [shared message contract](../types/lcm/README.md)
for field semantics and asynchronous real-data alignment.

The collector waits for a recorded READY probe before motion and a flushed terminal
event before stopping the logger. The auditor checks boundaries, counts, timestamps,
sample sequence continuity, observation validity, and decodable RGB from every declared
camera in each sample. It rejects duplicate or missing views. It permits
independent command/state rates and traffic outside episode boundaries. Training
conversion must align the asynchronous streams by their timestamps. Missing samples,
logger failure, or incomplete boundaries prevent training eligibility. Failed attempts
remain available for analysis.

Training conversion should select `eligible_for_training: true` episodes, revalidate
raw log integrity, and retain the full-rate actions. Cleanup additionally requires
task version 9, `policy_observation: "current_four_camera_rgb_only"`, and all four
valid RGB frames per sample. Target visibility is diagnostic and does not determine
training eligibility.

## Verification

```bash
uv run check
HUMANOID_RUN_SLOW_TESTS=1 uv run check
HUMANOID_RUN_RENDER_TESTS=1 uv run check
```

Full cleanup episodes across three seeds run in real time and take several minutes
each, so they are opt-in with `HUMANOID_RUN_SLOW_TESTS=1`. Ordinary checks retain
the focused navigation, physical grasp, scene, success-condition, and stack tests.
Set both environment variables to run all slow and rendered checks together.

The rendered suite includes a complete recorded episode with RGB decoding and raw
stream auditing, alongside seeded physical task completion and cancellation tests.
