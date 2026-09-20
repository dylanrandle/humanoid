**Triskel OSC tuning report — simulation, 20 September 2026**

Selected and applied a Triskel configuration for smooth teleoperation with bounded
tracking error. The final profile passed **73 nominal accelerated scenarios,
9 live-pipeline trials, and the unreachable-target recovery probe**. Hardware
verification is still pending. This is the best profile found in this bounded
study, not a claim of a global optimum.

The final live trials' worst per-trial p95 position error was
**3.80 mm**, below the 5 mm gate.
Their worst p95 publication interval was **34.73 ms**,
below 41.7 ms. All selected live trials passed the orientation, motion, clearance,
solve, and timing gates. No hardware drivers were started.

**Selected configuration**

| Parameter | Starting weights | Final |
|---|---:|---:|
| Tool position cost | 5 | 5 |
| Tool orientation cost | 1 | 0.72 |
| Damping cost | 0.1 | 0.3 |
| Soft low-acceleration cost | 0.03 | 0 |
| Manipulability cost | 0.005 | 0.03 |
| Manipulability regularization | 1e-6 | 1e-6 |
| Nominal controller rate | 30 Hz | 30 Hz |
| Commanded joint velocity / acceleration | 1 rad/s / 2 rad/s² | Same |
| Joint-position margin | 0 | 0.005 rad |
| Minimum self-collision distance | 5 mm | 5 mm |
| Collision safe-displacement gain | 0.01 | 0.01 |

The applied values are in [Triskel's configuration](src/humanoid/config/robot/triskel.py).
These are kinematic QP costs, not servo PID gains. The starting weights were already
using the new arm-only manipulability objective; this study does not compare
manipulability against the removed joint-centering objective.

The task remains the custom regularized log-volume implementation with Pink
4.0.0. Only arm columns contribute to its metric;
base, wheel, and gripper velocities stay locked in arm IK. Separate controllers
own those coordinates. The moving-base trials verified base-relative tool holding.
The selected costs are specific to this metric and controller rate.

**Procedure and decisions**

The [procedure](OSC_TUNING.md) gates every scenario before ranking smoothness:
position p95 ≤ 5 mm and maximum ≤ 15 mm; orientation p95 ≤ 2° and maximum ≤ 5°;
no solver/non-finite/limit/clearance failures; small ramps settled within 1 s.
Real-time publication must have p95 ≤ 41.7 ms and fewer than 1% of intervals above
50 ms. Clearance and limit checks use commanded and measured states. Numerical
tolerances are 1 µm for clearance and 1e-6 in the relevant joint-limit units.

Smoothness is the worst of tool high-frequency RMS / 1 mm and each arm joint's
high-frequency RMS / 0.25°. Analysis retains the diagnostic's 2 Hz detrending and
0.25 s derivative smoothing. Raw velocities and actual integration periods check
hard acceleration limits; filtered jerk is a separate comparison metric.

1. Repeated the original baseline three times, then repeated it after correcting
   timestep bookkeeping. Disabled manipulability and soft smoothing separately.
2. Screened 12 ablation/tool-cost/damping configurations, 14 smoothing/refinement
   configurations, and 6 manipulability configurations, each on the same six cases.
   Damping ranged from 0.03 to 0.5, smoothing from 0 to 0.3, and manipulability from
   0 to 0.03. The 0.72 orientation seed follows the 5 mm / 2° error-tolerance ratio.
3. Ran the full 42-case training matrix: XY/XZ/YZ, three sizes, fixed/tilting
   orientation, two anchors, plus ramps and holds. Damping 0.5 failed tracking in
   five cases, reaching 6.12 mm p95; it was rejected.
4. Manipulability 0.03 improved the arm's volume during a stationary nonsymmetric
   hold. Lower tested weights had little effect. With the intermediate damping
   0.4 profile, removing manipulability failed the 2° orientation gate on two
   extended-pose cases. Soft low-acceleration smoothing added no consistent benefit.
5. The damping 0.4 candidate passed accelerated validation but failed the live
   hardest-path tracking gate. After fixing the scheduler, compared 0.3, 0.35,
   and 0.38 in the real pipeline. Selected 0.3, which had the lowest worst live
   screening score and greater tracking margin. Froze this profile before running
   a fresh 13-case confirmation suite at two unused poses and a 7 s period.
6. Re-ran all 42 training and 18 original validation cases with the final profile,
   the fresh confirmation suite, repeated live comparisons, and regularization
   sensitivity at 1e-7 and 1e-5, including the near-singular unreachable-target
   probe. All final runs passed their applicable gates.

The first proposed folded anchor was already self-intersecting by 11.3 mm. It was
rejected during geometry preflight, before controller comparisons, and replaced
with a folded pose having 32.8 mm initial clearance. Both are recorded in
[anchor preflight](logs/osc_tuning/anchor_preflight.json).

**Measured results**

Each entry below is the worst per-scenario statistic, not a pooled percentile.
Baseline comparisons use the same corrected motion-limit implementation and joint
margin as the final profile.

| Suite / profile | Cases | Position p95, mm | Peak position error, mm | Orientation p95, ° | Worst smoothness score | Worst filtered command jerk RMS, rad/s³ |
|---|---:|---:|---:|---:|---:|---:|
| Training / baseline | 42 | 1.04 | 3.22 | 0.51 | 0.735 | 1.915 |
| Training / final | 42 | 2.28 | 3.19 | 0.81 | 0.687 | 1.533 |
| Original validation / baseline | 18 | 2.68 | 3.21 | 0.46 | 0.720 | 7.994 |
| Original validation / final | 18 | 3.05 | 3.60 | 1.11 | 0.637 | 6.964 |
| Fresh confirmation / final | 13 | 3.19 | 4.14 | 1.23 | 0.669 | 7.240 |

The final profile reduced the worst smoothness score by 6.6% on
training and 11.4% on the original validation set, with increased
tracking error that stayed within the declared bounds. Because live results drove
further tuning, the fresh confirmation set is the independent final check.

Across the 73 final nominal scenarios, minimum sampled collision clearance was
11.28 mm, versus the 5 mm requirement.
Every nominal ramp was already inside the settling band when its ramp ended and
remained there for the following second. The final 13 s nonsymmetric hold improved
scaled manipulability volume by 2.08% relative to the disabled-objective
hold from the identical initial state. This remains a soft objective, so slow
posture adjustment during a stationary tool hold is expected.

On the hardest live offset trajectory, three interleaved repeats per profile with
the corrected scheduler gave:

| Metric | Baseline | Final |
|---|---:|---:|
| Mean per-run smoothness score | 0.636 | 0.615 |
| Sample standard deviation of that score | 0.035 | 0.020 |
| Mean worst-joint command jerk RMS, rad/s³ | 0.909 | 0.757 |
| Maximum per-run p95 position error, mm | 1.88 | 3.80 |

The live RMS difference is small relative to the repeat spread. Mean command jerk
was 16.6% lower, but three repeats do not establish a precise
population-level improvement. Publication timing was the clearest live improvement.
The nine final-profile live trials included these repeats, the home figure eight,
two fresh poses, added delay, reference jitter, and concurrent base translation/yaw.

**Controller and runtime defects found during tuning**

- Pink stores previous displacement. Reusing it with a different integration
  period changed the implied previous velocity. A focused regression reached
  2.78 rad/s² despite a 2 rad/s² limit. OSC now re-expresses the previous velocity
  over the current period before constructing its smoothing/acceleration terms.
- Near a joint stop, the position bound and continuous stopping-distance bound
  could conflict with the acceleration bound. The original unreachable probe
  produced three solver failures and an abrupt 18 rad/s² commanded stop. The
  revised constraint reserves the current displacement plus braking distance,
  `v * dt + v² / (2 * a_max)`, uses a compatible full-step position bound, and
  keeps targets 0.005 rad inside the stops. The final probe had zero solver failures,
  stayed within the command limits, and recovered after 4.65 s
  when the reachable target returned. Its measured joint margin stayed at least
  4.76 mrad. Unreachable-target tracking error is
  deliberately excluded from nominal error gates.
- Long sleeps caused timing jitter even in an empty 30 Hz loop on this host.
  The shared loop now approaches deadlines using shorter sleeps without busy
  waiting. Initial live p95 publication intervals were roughly 42–44 ms; the
  final selected-profile trials stayed below
  34.73 ms, with no >50 ms intervals.

OSC now returns explicit solve diagnostics. The batch runner records them, checks
clearance with separate Pinocchio data, and verifies FK against MuJoCo body poses.
Simulation resets accept an explicit configuration. Controller and orchestrator
nodes accept an isolated LCM URL for reproducible live simulation tests.

**Scope and transfer limits**

Physics used MuJoCo 3.10.0 at 200 Hz, with the repository's
position-actuator model (`kp=1000`, damping ratio 1). Accelerated controller updates
use the first physics tick at each 30 Hz deadline, giving 30/35 ms intervals; live
nodes use measured wall-clock intervals and the corrected scheduler. Active OSC
integrates commanded state, preserving the production controller's behavior.

The simulator disables robot self-contact; clearance was checked geometrically
on all 135 configured pairs. The base uses ideal planar velocity actuators, not
wheel/ground contact dynamics. The 100 g payload, half/double damping, one-period
delay, and 6 Hz reference perturbation are explicit sensitivity assumptions,
not identified hardware uncertainty. FK agreement checks model consistency, not
hardware calibration. Simulated position servos do not reproduce the Feetech PID,
backlash, quantization, or measured bus timing. Simulated arm speed peaked at
3.34 rad/s despite bounded command
velocities: command limits do not guarantee physical peak-speed limits.

Verify the frozen configuration on hardware starting with slow ramps and holds,
then the same tracking matrix and base-relative hold. Record actual actuator
tracking, effort, timing, and clearance margins. The 0.005 rad reserve covers the
observed simulation overshoot; it is not a calibrated hardware safety margin.

**Artifacts and reproduction**

- [Procedure](OSC_TUNING.md) and [frozen experiment definitions](src/humanoid/config/osc_tuning.py).
- [Final config and compact summary](logs/osc_tuning/study_summary.json).
- [All accelerated trial metrics](logs/osc_tuning/all_trials.csv): 705 executions,
  including rejected candidates and repeats. Raw NPZ traces, per-trial JSON,
  initial states, config/model/source hashes, dependency versions, dirty source
  patches, and untracked sources are saved beside each experiment.
- [Final training](logs/osc_tuning/final_training/results.json),
  [original validation](logs/osc_tuning/final_validation/results.json), and
  [fresh confirmation](logs/osc_tuning/final_confirmation/results.json).
- [Live refinement](logs/osc_tuning/realtime_refinement.json) and
  [final live trials](logs/osc_tuning/final_live_summary.json). Each live folder
  includes its reproducer, native body snapshots, command/feedback traces, solve
  records, and per-node timing summaries.
- [Study scripts](logs/osc_tuning/reproduce_study). Earlier folders named
  `selected_*` retain intermediate candidates; `final_*` and the compact summary
  identify the frozen 0.3-damping profile.

Run a new network-free reproduction with a fresh output name:

```bash
uv run python -m humanoid.robots.utils.osc_tuning \
  --name reproduce_final --suite confirmation \
  --set tool_orientation_cost=0.72 --set damping_cost=0.3 \
  --set low_acceleration_cost=0 --set manipulability_cost=0.03 \
  --set joint_position_margin=0.005
```

Raw artifacts under `logs/` are local and gitignored; archive that directory when
sharing the study. The source report, runner, scenarios, and regression tests are
part of the working tree. **`uv run check` passed formatting, linting, static type
analysis, and all 938 tests.**
