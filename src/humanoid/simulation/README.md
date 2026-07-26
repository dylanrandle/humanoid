# Native MuJoCo simulation

The simulation package owns Humanoid's native, single-world digital twin. It is kept
independent of LCM so the same model-construction path can be reused by future scene or
training integrations.

## Boundaries

- `model.py` imports the selected robot URDF into an uncompiled `mujoco.MjSpec`, resolves
  package meshes, adds actuators and mimic constraints, and adds the ideal planar root
  used by mobile robots.
- `scene.py` composes that robot spec into a named environment. The `empty` scene keeps
  the previous robot-only behavior; `floor-and-cube` adds a floor and free pickup cube.
- `binding.py` resolves joints, actuators, and root coordinates by semantic name after
  compilation. MuJoCo and Pinocchio array ordering is never assumed to match.
- `engine.py` owns `MjModel`/`MjData`, actuator targets, reset, physics stepping, and
  conversion back to the Pinocchio-shaped `RobotState` expected by the existing stack.
- `nodes/robot/simulation.py` is the LCM boundary. In `sim` runtime it replaces the
  hardware driver, consumes `ROBOT/JOINT_COMMAND`, and publishes `ROBOT/STATE`.
- `visualizers/mujoco.py` owns the passive native viewer attached to the simulation
  engine's live `MjModel` and `MjData`.

The former teleport-and-integrate actuator simulator has been removed. Tests that need
an actuator boundary use explicit local test doubles; production simulation is always
native MuJoCo.

## Model semantics

The URDF remains shared with the Pinocchio controller. MuJoCo-specific actuator and
solver defaults are added programmatically to that source model. Both builders return
an `MjSpec` rather than only a compiled `MjModel`: `build_mujoco_spec()` provides the
robot alone, while `build_mujoco_scene()` composes the selected named environment.

Robot geoms currently collide with the cube but not with themselves or the floor. The
floor supports the cube. The planar mobile base does not model vertical dynamics, and
excluding robot-floor contacts avoids artificial forces from wheel-mesh penetration.
Before enabling self-collision, the disabled collision pairs in each robot's SRDF must
be translated to MuJoCo excludes so startup configurations do not generate artificial
self-contact forces.

The `floor-and-cube` cube is a 4 cm, 50 g object centered at `(x=0.3, y=0.0)` in world
coordinates. It starts with its bottom face on the floor and has a free joint, so it can
be pushed, lifted, and dropped. These values can be overridden through
`FloorAndCubeSceneConfig` without changing the robot model. The selected scene defaults
to `empty`. It can be changed from the operator console while the stack is stopped, or
set for a standalone process with `HUMANOID_MUJOCO_SCENE`.

Mobile robots use three internal slide/hinge joints for x, y, and yaw. The upstream OSC
already publishes these generalized velocities alongside hardware wheel velocities, so
the digital twin can drive the ideal base without changing LCM messages. Wheel joints
continue to receive and simulate their commanded angular velocities.

## Timing and safety

The default physics timestep is 1 ms. The simulation node publishes at 500 Hz and takes
two physics substeps per LCM tick. Position targets latch and hold; velocity targets are
zeroed by the same 250 ms stale-command policy used by the hardware driver.

Restarting the simulation process resets it to the configured Home preset. The engine
also exposes `reset()` directly for future simulation-control messages and integration
tests. On macOS, the simulation node transparently relaunches itself with MuJoCo's
`mjpython` executable so the native passive viewer can share the main thread correctly.
On Linux without a graphical display, or when the native window cannot initialize, the
viewer is skipped while physics and LCM state publication continue normally.
