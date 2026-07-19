# Humanoid

A Python framework for multi-robot control and teleoperation in simulation and on real hardware. Humanoid supports SO101, ElRobot, and Panda platforms.

## Features

- **Multi-robot support**: Compatible with SO101, ElRobot, and Panda robot platforms
- **Actuator control**: Runtime-independent joint control with optional typed Feetech hardware
- **Visualization**: Real-time robot visualization using MeshCat
- **LCM middleware**: Lightweight Communications and Marshalling for inter-process communication
- **Operational space control**: Advanced control algorithms for precise robot manipulation
- **Keyboard teleoperation**: Interactive control interface for testing and development
- **Oculus teleoperation**: VR controller-based teleoperation for intuitive robot control

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

The panel opens at [http://127.0.0.1:8765](http://127.0.0.1:8765). It supports:

- Switching between simulation and real-hardware runtimes while the system is stopped.
- Selecting the robot model while the system is stopped.
- Starting and stopping the main stack.
- Starting and stopping the Oculus and keyboard teleop nodes independently.
- Selecting Idle, Homing presets (Home or Rest), Oculus, or Keyboard control.
- Starting and stopping LCM recording from the console, with the active file and
  logger failures reported in the data panel. Each recording is stored as
  `logs/recording_<timestamp>/recording.lcm` alongside a `robot.json` snapshot of
  the robot configuration.
- Selecting a managed recording and replaying it through the simulation or real
  runtime. The console requires the selected robot and its complete configuration
  to match the saved snapshot. Real-hardware replay requires an explicit operator
  acknowledgement.

Starting the main stack launches the robot driver, controller, visualizer, logger, orchestrator, and homing policy. The server binds to localhost and only stops processes it launched. Real-hardware selection and startup require an acknowledgement enforced by the server, and stale browser configuration is rejected.

Replay publishes the recorded command and mode channels on the normal LCM network.
Stop any other robot stacks or standalone drivers before starting playback. The replay
selector reads complete recording bundles directly from `logs/`; no browser upload is
required.

Oculus device setup is documented in the [oculus_reader](https://github.com/rail-berkeley/oculus_reader) repository.

The hardware abstraction is documented in
[`src/humanoid/hardware/README.md`](src/humanoid/hardware/README.md).
Root-state estimators and estimation algorithms are documented in
[`src/humanoid/state_estimation/README.md`](src/humanoid/state_estimation/README.md).
Feetech-specific setup and maintenance commands live with the driver in
[`src/humanoid/hardware/actuators/feetech/README.md`](src/humanoid/hardware/actuators/feetech/README.md).

### Running Tests

After installing the optional development dependencies above, run the complete
verification suite with:

```bash
uv run check
```

This runs the same formatting, linting, type-checking, and test suite as CI. The
Pytest suite includes the Node.js UI tests. To run only Python and UI tests, use
`uv run pytest`; to run only the UI tests, use `npm test`.

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
│   ├── state_estimation/ # Root-state estimators and estimation algorithms
│   ├── types/            # Type definitions and LCM message types
│   ├── ui/               # Local operator console backend and frontend
│   ├── utils/            # General utilities
│   └── visualizers/      # Visualization tools (MeshCat)
├── tests/                # Unit tests
└── scripts/              # Utility scripts
```

## Development

### Code Quality

The project uses several tools to maintain code quality:

- **Ruff**: Linting and code formatting
- **Pytest**: Testing framework
- **Type checking**: Static type analysis with ty

Run the complete CI-equivalent verification suite:

```bash
uv run check
```

Run linting:

```bash
uv run ruff check .
```

Run type checking:

```bash
uv run ty check
```

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
