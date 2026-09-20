**Simulation procedure for tuning operational-space control**

Target Triskel in MuJoCo, with smooth teleoperation and bounded tracking error.
Select a configuration in simulation, then verify that fixed configuration on
hardware. The numerical acceptance targets below are proposed experiment settings,
not measured performance or hardware ratings.

The selection rule is: reject candidates that violate tracking or motion
requirements, then choose the smoothest remaining candidate on scenarios withheld
from the parameter search.

**1. Fix the experiment before changing costs.**

Record the robot/model revision, source revision and any uncommitted changes,
simulator timestep, actuator configuration, payload, initial joint configuration,
trajectory, feedforward setting, and random seed when disturbances are used. Keep
the servo parameters and simulation model fixed during the OSC search. In a
dedicated runner, put experiment defaults in `src/humanoid/config/` and shared
experiment types in `src/humanoid/types/`.

Use the frozen pre-study Triskel weights as the reference run. The selected
configuration and measured results are recorded in `OSC_TUNING_REPORT.md`:

| Parameter | Baseline value | Treatment during the main search |
|---|---:|---|
| `dt` | `1 / 30` s | Fixed nominal controller period |
| `tool_position_cost` | `5.0` | Fixed cost scale |
| `tool_orientation_cost` | `1.0` | Set from error tolerances, then hold fixed |
| `damping_cost` | `0.1` | Search first |
| `low_acceleration_cost` | `0.03` | Search after damping |
| `manipulability_cost` | `0.005` | Disable for initial tuning; search last |
| `manipulability_regularization` | `1e-6` | Fixed within one task implementation |
| `joint_velocity_limit` | `1.0` rad/s | Fixed simulation envelope |
| `joint_acceleration_limit` | `2.0` rad/s² | Fixed simulation envelope |
| `damping_mask`, `low_acceleration_mask` | `1.0` | Uniform initially |
| `avoid_collisions` | `True` | Enabled |
| `min_collision_distance` | `0.005` m | Fixed clearance requirement |
| `collision_safe_displacement_gain` | `0.01` | Fixed additional motion penalty |
| `solver` | `quadprog` | Fixed |

These baseline values are frozen in [the experiment config](src/humanoid/config/osc_tuning.py).
The selected values live in [Triskel's config](src/humanoid/config/robot/triskel.py). Confirm the motion envelope
against the intended payload before hardware validation. A search must not relax
limits or clearance to obtain a better score.

Keep the base stationary and hold the gripper during arm tuning. Use a fixed
controller rate and reference-publication rate. The diagnostic's `--rate` changes
the reference-publication rate; OSC's nominal rate comes from `config.dt`. The
running controller also uses a bounded measured integration timestep, which a
tuning runner must record or reproduce.

Freeze the manipulability implementation before its final tuning stage. The current
task uses regularized log-volume. A future Pink task on a reduced arm model would
use a different measure, so its costs require a new baseline. Initial tuning with
`manipulability_cost=0` can proceed independently of that migration.

**2. Declare the acceptance gates and ranking rule.**

Use these initial targets for nominal, reachable trajectories. Change them to suit
the application before searching, and document any later revision as a new
experiment.

| Measurement, evaluated separately for every nominal scenario | Proposed gate |
|---|---|
| Reference to measured-tool position error | p95 ≤ 5 mm; maximum ≤ 15 mm |
| Reference to measured-tool orientation error | p95 ≤ 2°; maximum ≤ 5° |
| Small pose-ramp recovery | Within 5 mm and 2° within 1 s after the ramp ends, and stays there for the following 1 s |
| IK execution | No solver failures, non-finite outputs, or incomplete runs |
| Joint position, commanded velocity and acceleration | Within configured bounds, using raw commands and the integration timestep |
| Clearance | No configured-clearance violations on either commanded or simulated measured configurations, within a documented numerical tolerance |
| Real-time controller publication at 30 Hz | p95 interval ≤ 41.7 ms; fewer than 1% of intervals exceed 50 ms; no stale-feedback termination |

Evaluate ramps and transitions as well as cruise motion; do not remove inconvenient
transients from the peak-error check. Check commanded and measured joint motion
separately. Command limits do not bound physical acceleration or prevent a lagging
robot from entering a configuration the command model did not predict.

Among feasible candidates, minimize the worst normalized high-frequency position
RMS across all scenarios, using both the tool and every arm joint. Proposed ranking
scales are 1 mm for tool motion and 0.25° for joint motion:

```text
smoothness_score = max over scenarios and arm joints of:
    tool_high_frequency_rms_m / 0.001
    joint_high_frequency_rms_rad / radians(0.25)
```

These are fixed normalization scales, separate from the acceptance gates. For
scores indistinguishable within baseline repeatability, prefer lower commanded
jerk, then the simpler configuration. Preserve the full tracking/smoothness
comparison so the choice is reviewable. A failed tracking gate cannot be
compensated by a better smoothness score.

Keep the offline analysis settings fixed: initially a 2 Hz shake cutoff and a
0.25 s derivative-smoothing window, matching the existing diagnostic. These
filtered derivatives are useful for ranking. Use unsmoothed command velocities
and controller timesteps for checking hard acceleration bounds. Treat frequencies
above the reliable telemetry bandwidth as unmeasured.

**3. Use reproducible training and validation scenarios.**

Reset simulated `q`, velocity, controller integration history, and reference state
before each scenario. Allow initial contacts and actuators to settle consistently.
Save the actual initial state. Run every candidate against the same initial states
and disturbance seeds. Returning to Idle alone does not restore an identical pose.

| Scenario | Purpose | Use |
|---|---|---|
| Current figure-eight matrix at an agreed central arm pose, 8 s period, fixed orientation | Establish translation tracking and vibration | Training |
| Same matrix with 10° orientation bias | Check position/orientation tradeoffs | Training |
| Same two cases at a second comfortable, nonsymmetric arm pose | Avoid tuning only for the home Jacobian | Training |
| 10 mm translation and 5° orientation ramps, separately, followed by a hold | Measure responsiveness, settling, and post-command drift | Training and final validation |
| Figure eights at an unused arm pose and a 6 s period | Check generalization to new posture and speed | Validation only |
| Extended and folded poses with independently checked feasible references | Check conditioning, limits, and clearance | Validation only |
| Payload/friction variations and an added reference delay of one controller period | Check sensitivity to identified or explicitly assumed model uncertainty | Validation only |
| Slow base translation/yaw while holding a base-relative tool target | Check the complete mobile system | Final integration test |

The existing diagnostic runs XY, XZ, and YZ planes at 1×, 1.5×, and 2× size, with
two loops per setting. Its default 40 × 20 mm dimensions therefore include paths
as large as 80 × 40 mm. Check the whole matrix's reachability at each anchor pose.

The network-free batch runner now covers fixed-anchor resets, pose ramps, holds,
model perturbations, reference delay/jitter, and concurrent base motion. The
original live diagnostic still provides the figure-eight matrix. Use the fixed
six-scenario screen to prune costs, then the full 42-scenario training suite before
18 withheld validation scenarios. Screening alone is not complete validation.

Maintain a separate robustness set for intentionally unreachable targets. Judge
those by bounded motion, solver behavior, and recovery when a reachable reference
returns. They cannot satisfy the nominal tracking-error bounds by definition.

**4. Establish the baseline and tune one parameter family at a time.**

Run the current configuration three times to measure repeatability. Then run an
ablation with manipulability disabled, followed by one with both manipulability
and soft low-acceleration smoothing disabled. Keep damping, motion limits, and
collision avoidance active. These runs reveal what each secondary task contributes.

Set the translation/orientation balance from the declared error tolerances. For
equally important errors, use `position_cost / orientation_cost =
orientation_tolerance_rad / position_tolerance_m`. The proposed 5 mm and 2°
tolerances give a ratio of about 7. With position cost fixed at 5, orientation cost
starts near 0.72. Compare that seed with the current value of 1.0.

Pink squares the cost-weighted residuals: doubling a task cost quadruples its
quadratic penalty. This makes multiplicative searches more informative than tiny
additive changes. The costs express preferences; they do not set an error bound
or directly specify settling time. See [Pink's task formulation](https://github.com/stephane-caron/pink/blob/main/pink/tasks/task.py).

| Stage | Candidate values | What is held fixed | Selection |
|---|---|---|---|
| Damping | `0.03, 0.06, 0.1, 0.2, 0.4` | Chosen tool costs; low acceleration = 0; manipulability = 0 | Best feasible tracking/smoothness region |
| Soft smoothing | `0, 0.01, 0.03, 0.1, 0.3` | Selected damping; manipulability = 0 | Lower vibration without violating tracking or settling gates |
| Joint refinement | Up to 3 damping × 3 smoothing values around the best pair | All other settings | Check interaction missed by the individual sweeps |
| Manipulability, current log-volume task | `0, 0.001, 0.003, 0.005, 0.01, 0.03` | Selected tracking/damping/smoothing and regularization | Smallest cost giving a repeatable dexterity improvement while passing the other gates |

These are bounded search ranges around the current configuration, not claimed
optimal values. Save all results, including failed candidates. Screen the grid
once, then repeat the best two or three candidates three times. Re-run the
baseline between blocks, and vary candidate order to expose timing or initialization
bias. On a deterministic simulation, repetitions measure scheduling variability;
they do not substitute for different model conditions.

If no candidate is feasible, inspect which gate fails. Widespread velocity or
acceleration saturation can mean the requested trajectory exceeds the selected
motion envelope. Fix an infeasible experiment or document a changed requirement
before restarting; do not hide that failure in the score.

High low-acceleration weights can introduce lag and oscillation, so retain damping
and the explicit acceleration limit throughout. [Pink's low-acceleration task](https://github.com/stephane-caron/pink/blob/main/pink/tasks/low_acceleration_task.py)

**5. Evaluate manipulability and parameter interactions explicitly.**

For manipulability candidates, log the task's measure and the singular values of
the arm tool Jacobian. Normalize translation and rotation by fixed reference
scales before comparing singular values across runs. Evaluate lower-tail minimum
singular value as well as volume: an improved product of singular values can hide
a worsened weakest direction. Record the normalization with the experiment.

Include a stationary target hold of at least 10 s at multiple arm poses. Some arm
motion may be useful while dexterity improves; continued drift without benefit,
tool-error violations, or oscillation is grounds to reject the setting. This
objective is a weighted preference, so tool tracking must still be measured.

For the current log-volume task, retain `manipulability_regularization=1e-6` during
the cost search. Afterwards, check sensitivity at `1e-7` and `1e-5`, including
near-singular poses. Strong sensitivity means the numerical treatment needs
review before accepting the tuning. Those values do not transfer directly to
Pink's unregularized Yoshikawa task.

Keep `collision_safe_displacement_gain=0.01` during the main search. It adds a
configuration-dependent penalty on motion as well as the separate collision
inequalities, so changes can invalidate the damping result. If telemetry shows
that penalty dominates, evaluate a separate gain/damping study and repeat all
clearance tests. [Pink's barrier objective](https://github.com/stephane-caron/pink/blob/main/pink/barriers/barrier.py)

Only introduce per-joint damping or smoothing masks when repeatable traces
identify a specific problematic joint. Tune those multipliers with the global
cost fixed, then repeat validation; a seven-dimensional mask search is unnecessary
for the first pass.

**6. Use the existing diagnostic correctly and close measurement gaps.**

Start the normal stack with Triskel and the simulation runtime selected, place it
at the recorded anchor configuration, and leave it in Idle. The following command
then performs the orientation-aware figure-eight matrix:

```bash
uv run python -m humanoid.robots.utils.controller_tracking \
  --robot triskel --hold-gripper \
  --width 0.04 --height 0.02 --period 8 \
  --orientation-bias-deg 10 --rate 30 --settle 3 \
  --shake-cutoff 2 --derivative-smoothing 0.25 \
  --label osc_reference_r1
```

For fixed-orientation runs, use `--orientation-bias-deg 0`. Keep velocity
feedforward enabled during the main search. A final matched run with
`--no-velocity-feedforward` can diagnose how much performance comes from
feedforward, but belongs to a separate comparison.

Parameter changes belong in the configuration loaded by the running controller;
the diagnostic does not push its imported config into an already running node.
Restart the controller stack after changing a candidate, reset to the recorded
anchor, and verify its startup config. A label alone does not change any gains.
Ensure the diagnostic process and controller run the same candidate revision.

For comparisons, add `--compare-to` with the exact prior run directory or
`metrics.json` path. Passing the whole `logs/tracking` directory chooses the latest
run, which may not be the intended reference. Preserve each run's metadata, raw
telemetry, metrics, timing CSV, and reports.

| Measurement | Current support |
|---|---|
| Reference-to-command FK and reference-to-measured FK errors | `tracking.csv` and `metrics.json`, overall and per setting |
| Joint tracking, high-frequency motion, filtered acceleration/jerk, settle motion | Native telemetry and smoothness reports |
| Command publication periods | `controller_timing.csv` and diagnostic log |
| Fraction near the acceleration limit | Filtered command-acceleration proxy; not actual QP active-set telemetry |
| Actual solve status, solve duration, integration timestep | Batch runner records explicit `ControlResult.diagnostics` and each integration timestep; the live LCM schema does not carry diagnostics |
| Minimum collision distance and minimum joint-limit margin | Batch runner checks commanded and measured configurations |
| Manipulability and scaled Jacobian singular values | Batch runner records log-volume and scaled arm Jacobian singular values |
| Pose-ramp settling time and fixed-anchor resets | Batch runner includes both |

The active [controller node](src/humanoid/nodes/robot/controller.py) integrates
commanded configurations rather than resynchronizing arm positions from feedback
on every tick. Preserve that behavior in the simulation experiment. Compare
reference → command FK, command → measured joints, and reference → measured FK.
Measured tool FK also shares the model's calibration assumptions; simulator body
poses are a useful additional ground-truth check when available.

OSC catches solve exceptions and can return zero velocity. Such a run may look
smooth and still finish the diagnostic. Its result now contains explicit solve
diagnostics, which the batch runner counts. Successful completion alone is not a
valid feasibility check. Missing required metrics are unknown, not passes.

Run a simulation-only experiment without starting the live stack:

```bash
uv run python -m humanoid.robots.utils.osc_tuning \
  --name baseline --suite screen --repeat 3 --set joint_position_margin=0.005
uv run python -m humanoid.robots.utils.osc_tuning \
  --name candidate --suite training \
  --set tool_orientation_cost=0.72 --set damping_cost=0.3 \
  --set low_acceleration_cost=0 --set manipulability_cost=0.03 \
  --set joint_position_margin=0.005
```

The runner starts from the frozen baseline in `config/osc_tuning.py`, applies the
listed overrides, and saves every trace as NPZ plus JSON metrics and source/model
hashes in `logs/osc_tuning/<name>`. Existing directories are never overwritten.
Use `--suite validation` only after freezing finalists; `--suite robustness`
contains an intentionally unreachable target followed by a recovery hold. The
`confirmation` suite reserves two additional poses and a 7 s period for independent
confirmation after live-pipeline results required one further damping refinement.

Physics remains at 200 Hz. Controller deadlines are quantized to physics ticks,
producing 30/35 ms integration intervals at a nominal 30 Hz. The controller now
rescales Pink's previous-displacement bookkeeping to the current timestep so its
velocity smoothing and acceleration constraint use the actual previous velocity.
The first baseline exposed this integration issue; preserve pre-fix and post-fix
baselines separately. The unreachable-target probe also exposed an incompatible
position/acceleration constraint near a stop. The final braking constraint requires
`v * dt + v**2 / (2 * a_max) <= remaining_joint_distance`, and the final Triskel
configuration reserves `joint_position_margin=0.005` rad inside the stops.
Apply that same margin to both baseline and candidate when reproducing the final
comparison; it was introduced after the cost search, followed by full revalidation. Finalists must also pass through the real-time nodes with
an isolated LCM URL. The study also found host timer-coalescing jitter: the shared
loop now approaches each deadline with shorter sleeps, without busy waiting.
Compare weights with the same scheduler; do not attribute scheduler improvements
to OSC costs. Each stage promotes only a few candidates to the next stage.

**7. Validate the finalist and prepare the hardware handoff.**

Evaluate the finalists on the withheld scenarios without changing their costs.
Use the same disturbance seeds for every finalist. Rank with the declared rule,
check every scenario separately, and retain a simpler candidate when the observed
improvement is within repeatability. If validation drives another tuning cycle,
reserve a new validation set so it remains an independent check.

Save the selected configuration, rejected candidates, acceptance thresholds,
training and validation results, model assumptions, and exact code state. Mark
the result as simulation-validated until hardware testing is complete.

On hardware, first repeat the short, slow scenarios with the selected configuration
and confirmed motion envelope, then the nominal tracking matrix and base-motion
integration test. Compare end-to-end errors, vibration, timing, and available
effort/thermal measurements. If command tracking is good but measured tracking
fails, investigate actuators, timing, payload, and model mismatch before changing
OSC weights. Any later parameter change creates a new candidate to validate.

Use one row per candidate and scenario in the experiment record:

| Run | Config revision | Scenario / initial state / seed | Tracking gates | Motion / clearance / solve gates | Tool HF RMS | Worst-joint HF RMS | Command jerk | Decision and reason |
|---|---|---|---|---|---|---|---|---|
| reference_r1 | recorded before execution | recorded before execution | pending | pending | pending | pending | pending | baseline |
