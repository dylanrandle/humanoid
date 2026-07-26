# Humanoid

A Python framework for multi-robot control and teleoperation in simulation and on real hardware. Humanoid supports SO101, ElRobot, and Panda platforms.

## Features

- **Multi-robot support**: Compatible with SO101, ElRobot, and Panda robot platforms
- **Actuator control**: Runtime-independent joint control with optional typed Feetech hardware
- **MuJoCo digital twin**: Native rigid-body dynamics behind the same LCM interface as hardware
- **Visualization**: Native MuJoCo simulation viewer plus MeshCat robot overlays
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
robot, runtime, and named MuJoCo scene; manage the stack and teleoperation processes;
choose a control mode; monitor node health; and record or replay LCM sessions.

Simulation uses MuJoCo and real-hardware mode uses the hardware driver; both expose the
same LCM interface to the rest of the stack. Hardware actions require explicit operator
acknowledgement. Stop other stacks or standalone drivers before replaying on the shared
LCM network.

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
│   └── visualizers/      # Native MuJoCo and MeshCat visualization
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
