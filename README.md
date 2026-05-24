# Humanoid

A Python framework for controlling small household humanoid robots with support for multiple robot platforms including SO101, ElRobot, and Panda.

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

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Install development dependencies** (optional):
   ```bash
   uv sync --group dev
   ```

## Usage

### Starting the Robot System

The main entry point starts three nodes in parallel:
- **Robot Driver**: Interfaces with hardware or simulation
- **Robot Controller**: Executes control algorithms
- **Robot Visualizer**: Provides real-time visualization

```bash
uv run start
```

Or using the Python module directly:
```bash
uv run python -m humanoid.start
```

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

### Keyboard Teleoperation

Control the robot using keyboard input:
```bash
uv run python -m humanoid.nodes.keyboard_teleop
```

### Oculus Teleoperation

Control the robot using Oculus VR controllers for intuitive, immersive teleoperation:
```bash
uv run python -m humanoid.nodes.oculus_teleop
```

**Device Setup**: For instructions on setting up your Oculus device, refer to the [oculus_reader](https://github.com/rail-berkeley/oculus_reader) repository.

### Running Tests

```bash
uv run pytest
```

## Project Structure

```
humanoid/
├── src/humanoid/
│   ├── controllers/       # Control algorithms (operational space, etc.)
│   ├── environment/       # Environment interfaces (realtime, base)
│   ├── middleware/        # Communication middleware
│   ├── motors/           # Motor drivers (Feetech, simulation)
│   ├── nodes/            # Main system nodes (driver, controller, visualizer)
│   ├── policy/           # Control policies (keyboard teleop, etc.)
│   ├── robots/           # Robot definitions and URDF assets
│   ├── types/            # Type definitions and LCM message types
│   └── visualizers/      # Visualization tools (MeshCat)
├── tests/                # Unit tests
└── scripts/              # Utility scripts
```

## Development

### Code Quality

The project uses several tools to maintain code quality:

- **Ruff**: Linting and code formatting
- **Pre-commit**: Git hooks for automated checks
- **Pytest**: Testing framework
- **Type checking**: Static type analysis with ty

Install pre-commit hooks:
```bash
uv run pre-commit install
```

Run linting:
```bash
uv run ruff check .
```

Run type checking:
```bash
uv run ty check
```

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
