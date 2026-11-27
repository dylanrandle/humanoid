# Unified robot backend

`uv run start --no-open` starts the same backend used by the dashboard, agents and scripts.
`--port` selects the HTTP listener port. `HUMANOID_API_URL` selects the Python client's
endpoint; managed agent and teleoperation processes inherit the running backend's address.
All HTTP operations use `/api`. The API stays available when the robot stack or motion
policy stops. [Motion and observation contract](../agent/README.md#http-contract)

## Clients

`api/clients/` contains callers of the unified HTTP API, mirroring the organization of
backend implementations in `api/services/`. `clients/robot.py` provides `RobotClient`
for agents and scripts; `clients/recording.py` provides the nonblocking `RecordingClient`
used by input nodes.

## Routes

`GET /api/logs` serves application logs. `GET /api/recordings` reports LCM recorder
status; `POST /api/recordings/start`, `/stop` and `/toggle` control LCM recording.

`api/routes/` groups HTTP route adapters by service area: agent tasks, application logs,
LCM recording, replay, runtime configuration/processes/modes, robot motion/observations,
and camera streaming. `routes/common.py` holds shared JSON parsing, safety-context and
success-response helpers. `server.py` assembles the application, registers these routes,
and owns request security, response headers, error handling, static serving and startup.
`config/api.py` centralizes endpoint defaults, server limits, dashboard assets, and security headers.
`ApiRoute` in `types/api.py` defines all HTTP paths for server adapters, Python clients
and generated browser constants, using `<name>` placeholders for path parameters.

## Ownership

`types/recording.py` defines `RecordingState`, `RecordingStatus`, and `RecordingOwner`
for LCM recordings. `types/logs.py` defines `LogEntry` and `LogSnapshot`; `config/logs.py` defines log settings. The
backend status payload exposes the current LCM recorder under `recording`.

| Component | Responsibility |
| --- | --- |
| `BackendApplication` | Construct services, aggregate status, supervise lifecycle and close resources |
| `RuntimeService` | Runtime/robot/scene selection, managed processes, replay exclusion and failure cleanup |
| `ModeService` | Mode requests, homing parameters and mode acknowledgement |
| `MotionService` | Validate requests, pin sessions/deadlines and correlate pipe acknowledgements/status/cancellation |
| `MotionExecutor` and `MotionPolicy` | Deduplicate, admit and execute bounded motions using fresh feedback |
| `ObservationService` | Read state and synchronized camera captures independently of motion execution |
| `RecordingService` | Own `lcm-logger`, episode ownership, finalization and recorder failure reporting |
| `LogService` | Collect parent/child diagnostic logs, rotate files and serve cursor-based log snapshots |
| `AgentTaskService` | Agent task admission and harness lifecycle |
| `StreamingService` | WebRTC signaling, video tracks, viewer limits and session cleanup |
| `telemetry/` | Host resources, node rates, actuator health and stream metrics |
| Orchestrator node | Switch modes, multiplex downstream commands and drain source queues during handoff |

The HTTP routes adapt typed values to these services. A future MCP adapter should call the
same service methods, including session, mode, recording ownership and hardware acknowledgement
checks. It should never call Flask route functions or publish controller commands directly.
MCP support itself is a follow-up; there is no MCP dependency or listener yet.

Backend supervision runs without browser polling. Recording has its own process monitor.
Stopping the stack finalizes recording before stopping nodes, invalidates motion sessions,
and closes observations and streams. Diagnostic logging remains available until API shutdown.
There is no logger node or orchestrator logging event/status channel.

`BackendApplication` constructs its own services. Its `runtime` and `robot` parameters
select the runtime and robot name; `lcm_url` selects the bus, including a private bus for
simulation trials. `create_observation_service(robot_config)` creates backend-owned
observations from a trial's custom configuration and closes any previous instance.
The backend passes `MotionService` through its `MotionProcessLifecycle` interface into
`NodeManager`. The process manager does not import API services. `AgentTaskService` receives
explicit status, configuration, observation, identity and mode capabilities rather than the
application object or its private attributes. It combines a reusable `AgentTask` instruction
with model, robot, mode and identity settings in `AgentExecution` before starting a worker.

`runtime_id` identifies the robot runtime; `motion_session_id` identifies motion authority.
Observation clients pin the runtime independently of the executor. Runtime resets invalidate
observations and episode recordings. Motion executor restarts invalidate only motion authority;
read-only clients and episode recordings continue. `RecordingOwner` contains `runtime_id` and
`episode_id`; an old owner's stop retry cannot affect another recording.

Oculus Y sends a nonblocking `POST /api/recordings/toggle`. The backend serializes this toggle
with other recording requests and chooses start/stop from actual recording state. Episode
ownership is enforced for toggles as well as explicit start/stop. The input worker receives
confirmed `RecordingStatus` and polls `GET /api/recordings` for dashboard changes and recorder failures.
`RECORDING_FEEDBACK_POLL_SECONDS` controls feedback polling. Pending toggles are not queued
repeatedly, and an ambiguous failed toggle is never retried automatically. Unavailable feedback
is represented as unknown, rather than optimistically changing the local recording flag.

## Motion pipe

`MotionPolicyNode` owns the policy loop, motion executor, and private pipe adapter in
one managed node. The manager explicitly launches `MotionPolicyNode` through
`MotionPolicyNode.run_connected()`, passing its connection and motion session ID.
Pipe requests are handled by a worker outside the control loop.
Both the API service and node use `Pipe[MotionEnvelope]` from `middleware/pipe.py`.
The generic pipe receives its Pydantic message type and `PipeConfig` explicitly;
session IDs, heartbeats, deadlines and disconnect actions remain motion responsibilities.

The backend creates a duplex multiprocessing pipe for each managed autonomy process and closes
the unused endpoint after spawning. Each endpoint has bounded input/output queues and dedicated
reader/writer workers. Requests and replies use validated JSON envelopes, not pickle. The private pipe and
public HTTP API both name motion authority `motion_session_id`. The policy
tick never performs pipe I/O. The executor owns authoritative motion outcomes and deduplication. Heartbeat replies also report readiness; observation availability alone does not mean the motion executor is ready.

`MOTION_PIPE_CONFIG` in `config/motion.py` supplies the generic pipe settings.
Its `capacity` bounds transport queues and concurrent API requests; queue overflow interrupts
the channel instead of silently dropping work. `max_bytes` bounds an encoded message,
`poll_seconds` controls worker polling, and `join_seconds` bounds worker shutdown waits.
`MOTION_PROTOCOL_CONFIG` supplies motion-specific timing. `request_timeout_seconds`
bounds acknowledgement waits; a timeout can leave an accepted motion running, so query
the same motion ID before retrying. `heartbeat_seconds` controls ping frequency,
`contact_timeout_seconds` controls disconnect detection, and `startup_timeout_seconds`
allows the spawned executor to initialize before its first reply.
`MotionServiceConfig.max_motions` bounds retained IDs; when full, restart the motion policy
to create a new session. IDs and deadlines stay pinned until that session ends.

On disconnect the motion node cancels queued/running motion, publishes the established hold
and zero-base action, requests release from API, then exits. This uses the normal
orchestrator route. Stale feedback still follows existing policy/controller behavior; a pipe
is not an independent hardware stop mechanism. Restarting the backend, policy, stack or
configuration invalidates the session. No previous motion is automatically replayed.

## Dashboard WebRTC

`GET /api/cameras` supplies camera names, generation, ICE configuration and stream metrics.
`POST /api/streams` accepts an SDP offer plus `camera`, `stream` (`rgb` or `depth`) and
`depth_far`. It returns `session_id`, answer `sdp` and `type`.
`DELETE /api/streams/<session_id>` closes that peer.

The browser creates a video receiver for each RGB/native-depth view. Native depth is colorized
for display; agent snapshots retain metric depth, intrinsics, timestamps and registration data.
Metadata polling does not fetch JPEG frames. The API shares latest-frame camera sources among
viewers; video encoding runs per peer. No media history is queued by the camera source.

`StreamingConfig.fps` caps source delivery rate, `max_width` bounds encoded dimensions, and
`max_sessions` bounds concurrent video peers (each RGB/depth view counts as one).
`negotiation_timeout` bounds signaling work; `idle_timeout` reaps offers that never connect.
Failed/closed peers, explicit removal, configuration changes and API shutdown release tracks
and connections. The dashboard closes streams when hidden, on range changes and on exit,
and renegotiates after a connection failure or backend generation change.

The dashboard reports Live only when the source is fresh and the browser has video
data and is playing. If autoplay is blocked, it exposes the video's play control.
Safari uses a canvas overlay to present decoded WebRTC frames because its video layer
can remain black despite successful decoding. The overlay paints at up to 15 FPS and
is stopped with its stream; other desktop browsers use the native video element.
Scene/stack resets during negotiation return a retryable conflict.
In simulation, `SimulationNode` renders and publishes camera frames; a separate
`CameraDriverNode` is only used for hardware cameras.

Camera streaming supports local-network viewing. Both the browser and backend use
local ICE candidates with no external discovery or relay servers. WebRTC still uses
ICE to negotiate the direct media connection.

The browser needs a direct media path to the backend, such as the same host or LAN.
HTTP remains loopback-bound with host/origin protections. A loopback SSH forward can
carry API calls and signaling, but does not tunnel WebRTC media.

Motion argument validation is shared with agent tools and trial programs through the
schemas in `types/motion.py`. `motion/validation.py` converts validated arguments to a
native `MotionTarget`; the policy checks runtime limits and produces per-tick `Action`
values. Model-specific response schema requirements stay in the agent harness.
