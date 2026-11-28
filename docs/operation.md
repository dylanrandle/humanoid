# Operating the robot

The operator console manages the robot stack, control inputs, and recordings.
Follow [Getting started](getting-started.md) to launch it.

## Console and control inputs

Select the robot and runtime before starting the stack. Simulation and hardware
share the same control interface; hardware actions require explicit operator
acknowledgement. Start the input you want to use, then select its control mode.
Use **Idle** to leave active control.

Keyboard and Quest provide teleoperation; [Agent control](agents.md) covers API mode.

## Quest teleoperation

Connect using [the local HTTPS gateway](gateway.md) or [Mac USB](hardware/quest.md#mac-usb).
The viewer includes offline assets for the
[Quest 3 Touch Plus controllers](hardware/quest.md#controller-assets).

1. Choose runtime, robot, and scene, then start the stack and Quest input.
2. Open the headset's connection page, choose **Open Vuer**, and enter VR or passthrough.
3. Release both grips, select Quest mode, then hold a grip to control.
4. Close the viewer before switching hosts. Stop input and stack when finished.

The right controller moves the tool, the left stick drives, and the right stick
turns. A/B closes/opens the gripper, X homes, and Y toggles recording. Each clutch
starts from the current tool target. Tracking loss pauses commands and requires
releasing both grips before control resumes. Physical controllers are required.

On Triskel, the viewer combines a large base-camera view with a wrist inset at its top-right.
Other robots use their [configured camera order](hardware/quest.md#architecture).
Unavailable sources show a label; calibrated observations remain unchanged.
[Quest configuration](../src/humanoid/config/quest.py) defines `base_height` for
panel size, `base_down_degrees` for placement below forward gaze, `wrist_fraction`
for inset size, and `wrist_margin` for inset padding. Its `input_timeout` controls
transport freshness. The [teleoperation configuration](../src/humanoid/config/teleop.py)
separately controls policy input freshness and motion filtering.

## Health and metrics

The **Health & metrics** panel groups motors, LCM channels, and nodes with host resources.
Motor feedback and temperatures appear only on hardware; simulation hides that subsection
without reserving space. The last driver or servo failure remains visible after stopping.
Host telemetry covers all processes on the backend machine, even with the stack stopped.

Use the LCM subsection to check observed publish rates and time since the last message,
including camera and custom channels. Quiet channels show zero Hz and an idle status;
event channels do not need to publish continuously. Application logs appear below the
health panel. See the [LCM rate monitor](../src/humanoid/telemetry/lcm.py) for measurement
and configuration details.

## Live 3D viewer

On macOS, open the native MuJoCo viewer alongside a running stack:

```bash
uv run viz
```

Use `uv run viz --host triskel` for a remote stack over SSH. The viewer follows
hardware, simulation, and replay. Amber shows commanded joints, blue shows the tool
target, and the arrow shows the base target and forward heading.

Simulation includes scene objects; Empty and hardware appear on a reference floor.
Both scenes use a white background and fixed overhead and lower fill lights that
approximate Viser's lighting. Empty also uses Viser's gray grid; Toy cleanup keeps
its scene floor. The native renderer uses ambient fill in place of HDR environment
reflections. Until feedback arrives, the viewer displays the configured home pose.
Drag to orbit, scroll to zoom, and close the window to disconnect.

## Cameras

The console streams RGB and colorized native depth for configured cameras.
The depth scale control adjusts the displayed range; black means no measurement.
Stale streams are labeled. See [Cameras](hardware/cameras.md) for depth interpretation
and [Networking](networking.md#camera-streaming) for remote video requirements.

## Record and replay

Use the console to record or replay LCM sessions. Each recording is a bundle under
`logs/recordings/<timestamp>/` containing the LCM log and robot configuration manifest.
The console reflects confirmed recorder state, including controller-triggered recording.

Stop other stacks or standalone drivers before replaying on the shared LCM network.
The viewer can follow replay just as it follows a live stack.

## See also

- [Documentation index](README.md)
- [Simulation scenes](simulation.md)
- [Deployment](deployment.md)
- [Controller diagnostics](diagnostics.md)
