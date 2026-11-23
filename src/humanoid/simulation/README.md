# Native MuJoCo simulation

The simulation package owns Humanoid's native, single-world digital twin. It is kept
independent of LCM so the same model-construction path can be reused by future scene or
training integrations.

## Boundaries

- `model.py` imports the selected robot URDF into an uncompiled `mujoco.MjSpec`, resolves
  package meshes, adds actuators and mimic constraints, and adds the ideal planar root
  used by mobile robots.
- `binding.py` resolves joints, actuators, and root coordinates by semantic name after
  compilation. MuJoCo and Pinocchio array ordering is never assumed to match.
- `engine.py` owns `MjModel`/`MjData`, actuator targets, reset, physics stepping, and
  conversion back to the Pinocchio-shaped `RobotState` expected by the existing stack.
- `nodes/robot/simulation.py` is the LCM boundary. In `sim` runtime it replaces the
  hardware driver, consumes `ROBOT/JOINT_COMMAND`, and publishes `ROBOT/STATE`.

The former teleport-and-integrate actuator simulator has been removed. Tests that need
an actuator boundary use explicit local test doubles; production simulation is always
native MuJoCo.

## Model semantics

The URDF remains shared with the Pinocchio controller. MuJoCo-specific actuator and
solver defaults are added programmatically to that source model. The builder returns an
`MjSpec` rather than only a compiled `MjModel`, allowing callers to attach the robot to
future MuJoCo scenes before compilation.

Robot geoms currently collide with external scene geoms but not with each other. Before
enabling self-collision, the disabled collision pairs in each robot's SRDF must be
translated to MuJoCo excludes so startup configurations do not generate artificial
self-contact forces.

Mobile robots use three internal slide/hinge joints for x, y, and yaw. The upstream OSC
already publishes these generalized velocities alongside hardware wheel velocities, so
the digital twin can drive the ideal base without changing LCM messages. Wheel joints
continue to receive and simulate their commanded angular velocities.

## Timing and safety

`MujocoSimulationConfig.physics_timestep` sets the duration of each physics step.
The simulation node publishes `ROBOT/STATE` at `MujocoSimulationConfig.publish_rate_hz`.
Each publish period must span a whole number of physics steps. Position targets hold
until updated. The watchdog zeros stale velocity targets after `command_timeout_seconds`.

Restarting the simulation process resets it to the configured Home preset. The engine
also exposes `reset()` directly for future simulation-control messages and integration
tests.

## Cameras

The model builder attaches each configured RGB and depth sensor to its URDF optical
frames (+X right, +Y down, +Z forward). Triskel carries a bottom-plate D435 and a
wrist D405. The D405 uses the RealSense STL and a nominal mount transform on
`gripper_base_link`. Measure the physical mount to calibrate that transform.
Panda and SO101 run without onboard cameras.

The simulation node publishes native RGB, native depth and depth registered to RGB.
Triskel's bottom-plate topics use `CAMERA/BASE/` and its wrist topics use `CAMERA/WRIST/`.
Each image carries its own intrinsics, frame ID and scale. Multiply depth samples by
`depth_scale` to get axial metres. Zero represents missing depth. Simulated pinholes
use nominal optics. Hardware publishes device calibration.

Each camera samples on its own simulation-time clock using `cameras[name].simulation.fps`
from the robot config. The same camera entry defines its optics.
Its worker renders the latest pending snapshot and publishes it
with the corresponding robot state's `timestamp`. The image's `device_timestamp` records
simulation time. If rendering falls behind, newer snapshots replace pending ones.
Passing `--record` to the agent trial CLI saves these sensor channels alongside
simulation state and robot commands in an LCM bundle for each episode.
Set `MujocoSimulationConfig.enable_cameras=False` to disable rendering for all sensors.

For camera preview transport and raw image subscriptions across hosts, see
[LCM networking](../../../README.md#lcm-networking).

Rendering needs an OpenGL context. Desktop macOS uses CGL; headless Linux can use
`MUJOCO_GL=egl` with suitable GPU drivers or `MUJOCO_GL=osmesa` with Mesa installed.
Deployment installs Ubuntu EGL/Mesa libraries and sets `MUJOCO_GL=egl` in the
systemd service. For a manual Ubuntu launch:

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libegl-mesa0 libopengl0 libgl1-mesa-dri
MUJOCO_GL=egl uv run start --no-open
```

Set the backend **before starting Python**, since MuJoCo selects it at import time.
The service user needs access to `/dev/dri/renderD*` (the `render` group on Ubuntu).
`MUJOCO_GL=osmesa` with `libosmesa6` is a slower software alternative on machines
without a compatible GPU. GLFW requires an X11 display and cannot render from a
headless systemd service with no `DISPLAY`.

## Cleanup scene and live viewer

Run `uv run start`, choose **Simulation**, and select **Toy cleanup** under
**Robot model** before starting the stack. The room contains an open bin, three
wheeled cars, three cubes, three cylinders, and three balls. Restarting restores
their starting positions. **Empty** is the default scene.

To view the running scene, see [Live 3D viewer](../../../README.md#live-3d-viewer).

## Floor pickup test

Panda and Triskel use each toy’s simulated pose to pick up a cube, cylinder,
ball, and car. Both fingers must support the toy at least 60 mm above its
starting height for 10 seconds, with less than 2 mm of height variation.
For Triskel’s cube grasp, finger penetration and upward motion before lift
must each stay below 1 mm.

```bash
uv run pytest -q tests/simulation/test_privileged_grasp.py
```

To save an MP4 and a synchronized 25 Hz CSV for each trial, prefix the command
with `HUMANOID_GRASP_OUTPUT=logs/privileged-grasp/run-1`.

### Privileged single-cube delivery

Run the pinch-point controller from the floor pickup test through approach, lift,
transport, release and retreat in randomized rooms. The base translates under
velocity control and the jaws carry the free cube through contact. The routine
reads the simulated cube, bin and pinch poses. It scores a delivery after a
one-second settling interval with the cube inside the bin, in contact with its
floor and clear of both fingers.

```sh
uv run python -m humanoid.simulation.privileged_cleanup --seeds 0 1 2 3 4 5 --video
```

The default closure is approximately 0.696 rad for the 36 mm cube. Pass
`--closure 0.6956521739130438 1.05 2.2028` to compare three jaw targets at
the same closing speed on each layout. Each trial saves a result and per-step
contact and pose trace under `logs/privileged-cleanup/`. With ffmpeg installed,
`--video` also saves the room view and a close-up tracking the cube.
