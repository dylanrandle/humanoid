# Agent trials

Trials let an agent explore a task, write a controller, and evaluate that controller
on new layouts. Start with simulation before running a hardware trial.

## Run toy cleanup

Connect ChatGPT through the [agent panel](agents.md#console-tasks), start Docker,
and run the trial from the repository root.

```bash
uv run python -m humanoid.agent.harness.trial --explore 3 --test 3 --record toy_cleanup --items 1
```

The agent explores three seeded MuJoCo layouts with one cube, then writes a Python controller.
The runner tests that controller in Docker on the next three seeds. A toy counts as deposited
when it settles inside the bin after release, supported by the bin floor or another deposited toy.

Place shared trial options before the task name. Use `--seed` to choose the first layout and
`--explore-seconds` or `--test-seconds` to set the per-episode time budgets.
Use `toy_cleanup --items 12` for the full toy set.
To evaluate a saved controller directly, pass `--replay PATH/TO/controller.py --seed N`,
where `N` is the first test layout.

**Hardware.** Start the real stack and motion policy, then add `--backend real --robot triskel`.
For each episode, set up the layout, home the robot, and select **API**. The runner
requests **Idle** before asking you to grade the result. For remote control and recording, follow [Networking](networking.md#remote-api-access).
Set `--endpoint` to the forwarded address.

## Review results

The runner prints the run directory under `logs/trials/`. Use `--output` to change it.

| File in the run directory | What to inspect |
| --- | --- |
| `explore-<seed>/index.html` | Each decision's intent, camera images, tool arguments, and result. Reload during a run or copy the whole episode folder to view it elsewhere. |
| `controller.py` | The generated controller used for evaluation and replay. |
| `test-<seed>.json` | Motion commands, measured outcomes, score, and the reason the episode ended (`ended_by`). |

`--record` saves one LCM log per episode, including camera and robot channels. The episode
JSON gives its path in `lcm_recording`. Simulation logs go under the run's `recordings/`
directory. Hardware logs stay on the robot. Every exploration gets an HTML review.

With FFmpeg installed, this command exports synchronized RGB and colorized depth MP4s from a log or bundle.

```bash
uv run python scripts/export_camera_videos.py /path/to/recording.lcm
```

## Add a simulation task

Implement `SimulationTask` with a scene, goal, and `score(data)` method. Override
`configure_scene(spec, seed)` for randomized layouts. Register the task's CLI argument
function in `TASKS`. [Toy cleanup example](../src/humanoid/simulation/tasks/toy_cleanup.py) · [Task registry](../src/humanoid/simulation/tasks/__init__.py)

## See also

- [Documentation index](README.md)
- [Agent control](agents.md)
- [Trial CLI](../src/humanoid/agent/harness/trial.py)
- [Trial contracts](../src/humanoid/types/trial.py)
