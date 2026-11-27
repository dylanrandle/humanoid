# Diagnostics

Use diagnostics to compare commanded motion with measured robot behavior.
Run commands from the repository root, including on a development machine connected
to the robot's LCM bus.

## Controller tracking

Run the diagnostic with the main stack in **Idle**:

```bash
uv run python -m humanoid.robots.utils.controller_tracking --robot triskel
```

The test starts from the current pose and commands a matrix of Cartesian figure-eight
motions through OSC/IK. It exercises translation, orientation, and optional gripper motion;
velocity feedforward is enabled by default. The robot returns to Idle when the test exits.

Each run is saved under `logs/tracking/controller_tracking_<robot>_<timestamp>/`. It includes
one report per trajectory setting plus raw telemetry and metrics that separate OSC error
(reference tool pose vs command FK) from joint and end-to-end tracking error.

Useful options:

- `--no-velocity-feedforward`: run a pose-only OSC comparison.
- `--hold-gripper`: disable gripper motion.
- `--label NAME --compare-to PATH`: label a run and compare it with prior results.
- `--help`: show trajectory, orientation, timing, and gripper options.

For example:

```bash
uv run python -m humanoid.robots.utils.controller_tracking \
  --robot triskel --label "P48 arm 2-4" --compare-to logs/tracking
```

## Simulated floor pickup

The grasp tests exercise pickup, holding, release, and contact behavior for Panda
and Triskel. Test code owns the exact scenarios and acceptance thresholds.

```bash
uv run pytest -q tests/simulation/test_privileged_grasp.py
```

Set `HUMANOID_GRASP_OUTPUT` to an output directory to save MP4s and synchronized
telemetry CSVs for the floor pickup trials.

## See also

- [Documentation index](README.md)
- [Simulation](simulation.md)
- [Controller tracking CLI](../src/humanoid/robots/utils/controller_tracking/cli.py)
- [Grasp tests](../tests/simulation/test_privileged_grasp.py)
