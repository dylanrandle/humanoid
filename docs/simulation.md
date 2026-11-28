# Simulation

MuJoCo provides a single-world digital twin behind the same LCM interface as
hardware. The simulation package itself is independent of LCM; the simulation
node connects it to the running stack.

## Scenes and interaction

Follow [Getting started](getting-started.md), then choose **Toy cleanup** under
**Robot model** before starting the stack. The scene contains a bin, cars, cubes,
cylinders, and balls. Restarting restores their initial positions; **Empty** is
the default scene. Use the [live viewer](operation.md#live-3d-viewer) to inspect it.

For pickup, approach from above with the finger pads on either side of the toy.
Grasp cars across the narrow chassis width. Close with `[` on the keyboard or
hold **A** with Oculus teleop engaged. Keep the grip target while lifting, then
open with `]` or **B**.

## Models and limits

The builder imports the robot URDF shared with the controller, then adds MuJoCo
actuators, constraints, and solver settings. Semantic joint names connect MuJoCo
state to the rest of the stack. Simulation gains are separate from hardware PID gains.

The configured wheel `floor_frame` defines the ground height for both the viewer's
reference floor and the cleanup scene. Scene objects are positioned relative to that
floor; robots without wheels use world zero. Keep the floor frames aligned with the
wheel bottoms in both Triskel URDFs when changing the base geometry.

Robot geometry collides with the scene, but self-collision is disabled. Mobile
bases use ideal planar motion, and finger compliance is approximated through
contact settings. Treat successful simulated manipulation as a model result,
not proof of hardware performance.

For Triskel, actuator dynamics reference the
[Feetech datasheet](https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf)
and [BAM bench fit](https://github.com/Rhoban/bam/blob/279c4828c466e7e19b631c45203b0e3582be9869/bam/params/feetech_sts3215_7_4V/m1.json);
finger geometry comes from the [TRI Fin Ray model](https://github.com/ToyotaResearchInstitute/lbm_eval/releases/tag/1.1.0).
When hardware changes, update both the
[simulation URDF](../src/humanoid/robots/assets/triskel/urdf/triskel.urdf) and
[planner collision URDF](../src/humanoid/robots/assets/triskel/urdf/triskel_collision.urdf).

## Timing and configuration

`MujocoSimulationConfig.physics_timestep` controls physics-step duration and
`publish_rate_hz` controls state publication. Each publish period must contain a
whole number of physics steps. `command_timeout_seconds` controls when stale
velocity targets are zeroed; position targets hold until updated. Restarting
the simulation resets it to the configured Home preset.

See [simulation defaults](../src/humanoid/config/simulation.py),
[configuration types](../src/humanoid/types/simulation.py), and
[robot definitions](../src/humanoid/config/robot/) for model and drive settings.

## Cameras

Configured sensors render from URDF optical frames. `cameras[name].simulation.fps`
controls each camera's simulation-time sampling; `MujocoSimulationConfig.enable_cameras`
controls whether sensors render. If rendering falls behind, newer snapshots replace
pending ones. See [Cameras](hardware/cameras.md) for depth semantics.

`cameras[name].simulation.render_shadows` controls shadows in sensor RGB images.
Disabling it reduces software-rendering cost without changing depth geometry.

Triskel's wrist-camera mount is nominal; calibrate its physical transform before
relying on it for hardware measurements. Panda and SO101 have no onboard cameras.

## Headless rendering

Rendering requires OpenGL. Desktop macOS uses CGL. Deployment installs EGL/Mesa
libraries and selects EGL for the service; for a manual Ubuntu launch:

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libegl-mesa0 libopengl0 libgl1-mesa-dri
MUJOCO_GL=egl uv run start --no-open
```

Set `MUJOCO_GL` before Python starts. The service user needs access to
`/dev/dri/renderD*`, usually through the `render` group. `MUJOCO_GL=osmesa` with
`libosmesa6` is a slower software alternative; GLFW needs an X11 display.

## See also

- [Documentation index](README.md)
- [Agent trials](trials.md)
- [Grasp diagnostics](diagnostics.md#simulated-floor-pickup)
- [Model builder](../src/humanoid/simulation/model.py) and [engine](../src/humanoid/simulation/engine.py)
