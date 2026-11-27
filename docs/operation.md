# Operating the robot

The operator console manages the robot stack, control inputs, and recordings.
Follow [Getting started](getting-started.md) to launch it.

## Console and control inputs

Select the robot and runtime before starting the stack. Simulation and hardware
share the same control interface; hardware actions require explicit operator
acknowledgement. Start the input you want to use, then select its control mode.
Use **Idle** to leave active control.

The console shows node health, motor feedback, temperatures, and application logs.
Host telemetry covers all processes on the backend machine, even with the stack stopped.
The last driver or servo failure remains visible after stopping.

Keyboard and Oculus provide teleoperation; [Agent control](agents.md) covers API mode.
For headset installation, see [oculus_reader](https://github.com/rail-berkeley/oculus_reader);
for a wireless connection, see [Deployment](deployment.md#wireless-oculus).
Each Oculus clutch starts from the current tool target. Commands pause when tracking
is lost or the latest frame exceeds `input_timeout`.

## Live 3D viewer

On macOS, open the native MuJoCo viewer alongside a running stack:

```bash
uv run viz
```

Use `uv run viz --host triskel` for a remote stack over SSH. The viewer follows
hardware, simulation, and replay. Amber shows commanded joints, blue shows the tool
target, and the arrow shows the base target and forward heading.

Simulation includes scene objects; hardware appears on a reference floor. Until
feedback arrives, the viewer displays the configured home pose. Drag to orbit,
scroll to zoom, and close the window to disconnect.

## Cameras

The console streams RGB and colorized native depth for configured cameras.
The depth scale control adjusts the displayed range; black means no measurement.
Stale streams are labeled. See [Cameras](hardware/cameras.md) for depth interpretation
and [Networking](networking.md#camera-streaming) for remote video requirements.

## Record and replay

Use the console to record or replay LCM sessions. Each recording is a bundle under
`logs/recordings/<timestamp>/` containing the LCM log and robot configuration manifest.
Oculus **Y** toggles recording; the console reflects confirmed recorder state.

Stop other stacks or standalone drivers before replaying on the shared LCM network.
The viewer can follow replay just as it follows a live stack.

## See also

- [Documentation index](README.md)
- [Simulation scenes](simulation.md)
- [Deployment](deployment.md)
- [Controller diagnostics](diagnostics.md)
