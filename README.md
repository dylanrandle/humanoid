# Humanoid

A Python framework for multi-robot control and teleoperation in simulation and on real hardware. Humanoid supports SO101, ElRobot, and Panda platforms.

## Features

- **Multi-robot support**: Compatible with SO101, ElRobot, and Panda robot platforms
- **Motor control**: Support for Feetech servo motors with configuration utilities
- **Visualization**: Real-time robot visualization using MeshCat
- **LCM middleware**: Lightweight Communications and Marshalling for inter-process communication
- **Operational space control**: Advanced control algorithms for precise robot manipulation
- **Keyboard teleoperation**: Interactive control interface for testing and development
- **Oculus teleoperation**: VR controller-based teleoperation for intuitive robot control

## Requirements

- Python 3.13 or higher
- [Node.js](https://nodejs.org/) 22 or higher (required for UI tests)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. **Install uv** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Node.js 22 or higher** for UI development and tests. The expected major
   version is recorded in `.node-version` for compatible version managers.

3. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd humanoid
   ```

4. **Install dependencies**:

   ```bash
   uv sync
   ```

5. **Install development dependencies** (optional):
   ```bash
   uv sync --group dev
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

### Motor Utilities

The package includes several utilities for working with Feetech servo motors:

**Scan for connected motors**:

```bash
uv run python -m humanoid.motors.feetech.scripts.scan
```

**Jog a motor**:

```bash
uv run python -m humanoid.motors.feetech.scripts.jog
```

**Set motor ID**:

```bash
uv run python -m humanoid.motors.feetech.scripts.set_id
```

**Zero motor position**:

```bash
uv run python -m humanoid.motors.feetech.scripts.zero
```

**Configure motor gains**:

```bash
uv run python -m humanoid.motors.feetech.scripts.set_gains
```

### Running Tests

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
│   ├── middleware/       # Communication middleware
│   ├── motors/           # Motor drivers (Feetech, simulation)
│   ├── nodes/            # Process-managed nodes grouped by domain
│   ├── orchestrator/     # Control-mode client, monitoring, and service
│   ├── policy/           # Control policies (keyboard teleop, homing, oculus, etc.)
│   ├── robots/           # Robot definitions and URDF assets
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

Robot and system configuration can be customized through the configuration system. See [`config.py`](src/humanoid/config.py) for available options.

## Troubleshooting

For common issues and solutions, see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
