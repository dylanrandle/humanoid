# Getting started

Run a local simulation to try Humanoid without connecting physical hardware.

## Install

Use Python 3.12, selected by `.python-version`, and the
[uv package manager](https://github.com/astral-sh/uv). From a checkout of this repository:

```bash
uv sync
uv run start
```

The console opens at [localhost:8765](http://127.0.0.1:8765).
Use `--no-open` to start without opening a browser; `--port` selects the listener port.

## Run a simulation

1. Select a robot and the **Simulation** runtime. Triskel includes cameras and a mobile base.
2. Start the main stack and wait for **Idle**.
3. Start a control input and select its mode to control the robot.
4. Return to **Idle** when finished, then stop the stack.

Continue with [Operating the robot](operation.md) for the viewer, recording, and replay,
or [Agent control](agents.md) for a small scripted movement.

## See also

- [Documentation index](README.md)
- [Development setup](development.md)
- [Hardware setup](hardware/README.md)
- [Troubleshooting](troubleshooting.md)
