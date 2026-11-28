# Robot backend

The console, agents, and scripts share one backend in `humanoid.api`.
Its HTTP listener remains available when the robot stack or motion policy stops.

## Service boundaries

Routes adapt HTTP requests to typed services. The backend owns runtime selection,
process supervision, modes, observations, motions, recordings, logs, and camera
streaming. Supervision continues independently of browser polling. The backend
also hosts the [Quest device integration](hardware/quest.md#architecture).

The motion policy runs in a managed process. A private pipe connects it to the
backend, with I/O workers keeping HTTP and model calls out of the control loop.
New integrations should use the services and their admission checks rather than
publish controller commands directly.

## Runtime and motion lifecycle

`runtime_id` identifies the robot runtime; `motion_session_id` identifies the motion
executor. A runtime reset invalidates observations and episode recordings. An
executor restart invalidates motion authority while read-only clients and episode
recordings continue. Recreate a client that has pinned authority before controlling
a replacement executor.

Stopping the stack finalizes recordings, stops nodes, and closes observations and
streams. Logs remain available until backend shutdown. Pipe disconnects cancel
outstanding motion and request Idle; this is not an independent hardware stop.

## Source entry points

Use [Agent control](agents.md) for a Python example. Exact routes, payloads, limits,
and transport behavior are defined in code:

| Source | What it defines |
| --- | --- |
| [Application](../src/humanoid/api/application.py) and [services](../src/humanoid/api/services/) | Service ownership and lifecycle |
| [Routes](../src/humanoid/api/routes/) and [API types](../src/humanoid/types/api.py) | HTTP adapters and paths |
| [Clients](../src/humanoid/api/clients/) | Robot and recording client interfaces |
| [Motion types](../src/humanoid/types/motion.py) | Shared validation and motion contracts |
| [Motion configuration](../src/humanoid/config/motion.py) | Pipe bounds and protocol timing |
| [Streaming configuration](../src/humanoid/config/streaming.py) | Video delivery, peer limits, and timeouts |

## See also

- [Documentation index](README.md)
- [Architecture](architecture.md)
- [Networking](networking.md)
