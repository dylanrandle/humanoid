# Humanoid

A Python framework for multi-robot control and teleoperation in simulation and on real hardware. Humanoid supports SO101, Triskel, and Panda platforms.

## Features

- **Multi-robot support**: Compatible with SO101, Triskel, and Panda robot platforms
- **Actuator control**: Runtime-independent joint control with optional typed Feetech hardware
- **MuJoCo digital twin**: Native rigid-body dynamics behind the same LCM interface as hardware
- **Visualization**: Real-time robot visualization using MeshCat
- **LCM middleware**: Lightweight Communications and Marshalling for inter-process communication
- **Operational space control**: Advanced control algorithms for precise robot manipulation
- **Teleoperation**: Keyboard and Oculus control interfaces

## Requirements

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. **Install uv** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd humanoid
   ```

3. **Install the Python dependencies**:

   ```bash
   uv sync
   ```

4. **Install development dependencies** (optional):

   Node.js and `npm ci` are only required when developing the web UI, formatting
   static assets, or running the complete verification suite. The expected Node.js
   major version is recorded in `.node-version` for compatible version managers.

   ```bash
   uv sync --group dev
   npm ci
   ```

## Usage

### Operator Console

The project is operated through a local web UI. Start it with:

```bash
uv run start
```

The panel opens at [http://127.0.0.1:8765](http://127.0.0.1:8765). Use it to select the
robot and runtime, manage the stack and teleoperation processes, choose a control mode,
monitor node health, and record or replay LCM sessions.

Simulation uses MuJoCo and real-hardware mode uses the hardware driver; both expose the
same LCM interface to the rest of the stack. Hardware actions require explicit operator
acknowledgement. Stop other stacks or standalone drivers before replaying on the shared
LCM network.

### Running on Triskel

Sync the current working tree, including uncommitted changes, to the robot:

```bash
./scripts/sync-code
```

The default destination is `triskel:~/humanoid`. Set
`HUMANOID_ROBOT_HOST` or `HUMANOID_ROBOT_DIR` to override it, and pass `--dry-run`
to preview the transfer. The command mirrors source files with rsync while preserving
remote Git metadata, environments, dependencies, caches, and logs. Other remote source
files that are absent locally are deleted after a successful transfer.

On the robot, install dependencies if needed and start the console without trying to
open a remote browser:

```bash
cd ~/humanoid
uv sync
uv run start --no-open
```

In another local terminal, forward the loopback-only dashboard and MeshCat ports:

```bash
./scripts/forward-ports
```

Then open [the dashboard](http://127.0.0.1:8765) and
[the visualizer](http://127.0.0.1:7000/static/). The tunnel defaults to dashboard port
`8765` and MeshCat port `7000`; its `--help` output lists environment variables for
overriding local or remote ports. MeshCat may select a higher remote port when `7000`
is already occupied, in which case set `HUMANOID_VISUALIZER_PORT` to the URL's port.

#### Wireless Oculus on Triskel

Enable Developer Mode and USB debugging on the Quest, connect it once over USB to a
computer with `adb`, and accept the headset's debugging prompt. Switch the headset's ADB
daemon to the standard wireless port and find its Wi-Fi address:

```bash
adb devices -l
adb tcpip 5555
adb shell ip route
```

Use the IPv4 address shown after `src`. The Quest and Triskel must be reachable on the same
network. Verify the connection from Triskel; put on the headset and accept its authorization
prompt if one appears:

```bash
ssh triskel 'adb connect 192.168.1.42:5555 && adb devices -l'
```

Replace the example address, then start the remote console with wireless Oculus enabled:

```bash
ssh triskel
cd ~/humanoid
uv run start --no-open --oculus-ip 192.168.1.42
```

`HUMANOID_OCULUS_IP=192.168.1.42 uv run start --no-open` is equivalent. With neither option
set, Oculus teleoperation retains USB discovery. The reader installs or starts its bundled
headset app automatically. It fails startup if no controller frames arrive within 15 seconds,
and input older than 300 ms is treated as disengaged so a dropped wireless stream holds the
current robot pose. Wireless ADB may need to be enabled again after the headset reboots.

## Project Structure

```
humanoid/
├── src/humanoid/
│   ├── controllers/      # Control algorithms (operational space, etc.)
│   ├── environment/      # Environment interfaces (realtime, base)
│   ├── hardware/         # Simulated and real device interfaces
│   ├── middleware/       # Communication middleware
│   ├── nodes/            # Process-managed nodes grouped by domain
│   ├── orchestrator/     # Control-mode client, monitoring, and service
│   ├── policy/           # Control policies (keyboard teleop, homing, oculus, etc.)
│   ├── robots/           # Robot definitions and URDF assets
│   ├── simulation/       # Native MuJoCo model, bindings, and physics engine
│   ├── state_estimation/ # Root-state estimators and estimation algorithms
│   ├── types/            # Type definitions and LCM message types
│   ├── ui/               # Local operator console backend and frontend
│   ├── utils/            # General utilities
│   └── visualizers/      # Visualization tools (MeshCat)
├── tests/                # Unit tests
└── scripts/              # Utility scripts
```

## Documentation

- [Hardware integrations](src/humanoid/hardware/README.md)
- [Actuator hardware](src/humanoid/hardware/actuators/README.md)
- [Feetech actuator setup and maintenance](src/humanoid/hardware/actuators/feetech/README.md)
- [Native MuJoCo simulation](src/humanoid/simulation/README.md)
- [State estimation](src/humanoid/state_estimation/README.md)

Oculus device setup is documented in the
[oculus_reader](https://github.com/rail-berkeley/oculus_reader) repository.

## Development

### Running Tests

After installing the optional development dependencies above, run the complete
verification suite with:

```bash
uv run check
```

This runs the same formatting, linting, type-checking, and test suite as CI. The
Pytest suite includes the Node.js UI tests. To run only Python and UI tests, use
`uv run pytest`; to run only the UI tests, use `npm test`.

### UI Constants

Python enums, payload keys, process mappings, and API routes are the source of truth for
the browser constants. After changing one of those definitions, regenerate the JavaScript
module:

```bash
uv run python scripts/generate_js_constants.py
```

The Python and UI test suites both verify that the generated module is current.

### LCM Types

LCM message types are auto-generated from `.lcm` definitions in `src/humanoid/types/lcm/`. To regenerate:

```bash
./scripts/generate-lcm-types
```

To inspect LCM messages:

```bash
./scripts/run-lcm-spy
```

## Configuration

Robot and system configuration can be customized through the configuration package. Robot definitions live in [`src/humanoid/config/robot/`](src/humanoid/config/robot/), with shared selection and visualizer settings alongside them.

## Troubleshooting

For common issues and solutions, see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
