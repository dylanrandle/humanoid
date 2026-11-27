# Architecture

Humanoid separates operator interaction, control, and device access so the same
robot workflows can run in MuJoCo or on hardware.

## How the stack fits together

The console and agents call a shared HTTP backend. The backend manages processes
and control modes. Policies produce targets, the orchestrator routes the active
policy's commands, and controllers translate them into robot commands. Hardware
or simulation returns feedback over LCM.

| Area | Responsibility |
| --- | --- |
| [API](api.md) and [UI](../src/humanoid/ui/) | Operator console, observations, motion requests, and lifecycle management |
| [Agents](agents.md) and [policies](../src/humanoid/policy/) | Task execution and control inputs |
| [Orchestrator](../src/humanoid/orchestrator/) and [nodes](../src/humanoid/nodes/) | Control-mode routing and managed processes |
| [Controllers](../src/humanoid/controllers/) and [robots](../src/humanoid/robots/) | Control algorithms, kinematics, and robot assets |
| [Hardware](hardware/README.md) and [simulation](simulation.md) | Physical devices and the MuJoCo digital twin |
| [State estimation](state-estimation.md) | Feedback-derived base pose and velocity |
| [Perception](../src/humanoid/perception/) | Segmentation, depth completion, and saved observations |
| [Middleware](../src/humanoid/middleware/) and [types](../src/humanoid/types/) | Communication and shared data contracts |

## Configuration

Concrete defaults live in [config](../src/humanoid/config/), including
[robot definitions](../src/humanoid/config/robot/). Shared configuration shapes
and message types live in [types](../src/humanoid/types/).

A robot definition selects its model, joint drives, cameras, and physical hardware.
The runtime selects hardware or simulation. Follow the relevant subsystem guide
for configuration entry points; the code defines individual fields and defaults.

## See also

- [Documentation index](README.md)
- [Development](development.md)
- [LCM networking](networking.md)
