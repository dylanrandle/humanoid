# Collection messages

Each `.lcm` file defines one wire type. Run `scripts/generate-lcm-types` after changing
schemas to regenerate the Python bindings and Java viewer types.

The following messages and their ordinary `Publisher`/`Subscriber` topics work in
simulation and real teleoperation:

| Type | Topic | Contents |
| --- | --- | --- |
| `camera_frame_t` | `CAMERA/FRAME` | Capture time, sensor/optical-frame identity, rectified RGB JPEG, dimensions, intrinsics |
| `object_state_t` | `OBJECT/STATE` | Measurement time, object identity, reference frame, pose, optional bounding box and velocity |
| `task_annotation_t` | `TASK/ANNOTATION` | Time, episode ID, task ID/version, task-specific JSON labels |
| `collection_episode_t` | `COLLECTION/EPISODE` | Episode lifecycle, task ID/version, runtime, clock, cumulative counts and metadata |

`CollectionEpisode.runtime` is `sim` or `real`. Object poses in simulation come from
ground truth; real recordings can carry pose estimates. There are no per-message
source fields or ground-truth flags. Task labels are shared across runtimes and their
JSON schema belongs to the named task/version. The episode identifies the task even
when no per-frame annotations are available. Simulation seeds, task configuration,
provenance, and reasons for success/failure belong in episode metadata; no seed is
required for a real episode.

Timestamps are nanoseconds on the wire and seconds in Python. All producers in an
episode must use the clock named in its boundary messages. Simulation collection uses
`host_monotonic`, matching the independent robot processes. Real collection can use
the same clock; hardware camera timestamps must be mapped onto that same clock.
Timestamp images at capture and poses at measurement, not at delivery. LCM event
timestamps record receipt and must not replace sensor timestamps during alignment.

Camera frames contain RGB only and require no world pose. Images must be rectified
before publication, with intrinsics for the stored resolution. `frame_id` names the
optical frame (x right, y down, z forward); `name` identifies the sensor when several
cameras share the topic. Object poses transform object-local coordinates into their
named reference frame; quaternions are wxyz. Bounding-box centers and half-sizes use
object-local meters, and velocities use reference-frame axes. Missing bounding boxes
or velocities use `None` in Python and explicit presence flags on the wire, so missing
estimates are not confused with zero velocity or zero-sized objects.

Episode outcomes are `ready`, `started`, `succeeded`, `failed`, and `aborted`. Terminal
events close an episode and carry its final counts and outcome evidence. Counts are
cumulative, with `-1` meaning unavailable/not applicable (for example, a real raw
stream with no bundled samples). A success event alone does not establish
recording integrity or training eligibility.

The simulation collector embeds the shared camera, object, and task messages in
`simulation_sample_t` to describe one received simulator snapshot. The last applied
command retains its application timestamp; `action_index = -1` denotes asynchronous
alignment. Episode action counts describe expert policy targets. Its camera wrapper adds
privileged world-from-optical extrinsics. It does not duplicate these large image
payloads on standalone topics. The simulation audit also requires complete ground-truth
geometry/velocity, monotonic sensor timestamps, and continuous sample sequences.
Episode metadata declares `camera_names`; every bundled sample must contain exactly
one RGB observation from each declared camera, with no missing or unexpected views.

A real teleoperation collector can instead record the standalone shared topics plus
the existing robot state and applied-command streams through `RobotLoggerNode` and
`lcm-logger`, then align their asynchronous observations in the training converter.
Real camera/estimator producers, episode controls, and real-specific integrity checks
remain to be integrated; the simulation runner and its auditor remain simulation-only.
