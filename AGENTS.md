# Python Execution Rules

- Always use `uv run` to execute Python scripts or tools.
- Example: Use `uv run python app.py` instead of `python app.py`.
- Example: Use `uv run pytest` instead of `pytest`.

# Verification and Code Quality Rules

Before submitting major changes, run the complete project verification suite. You do not need to run it for minor changes or small tweaks:

```bash
./triskel check
```

The command runs formatting checks, linting, static type analysis, and the complete test suite. To apply automatic formatting or lint fixes while working, use:

```bash
uv run ruff format
uv run ruff check --fix
```

# Project Organization Rules

- ROS package resources are the only source of physical robot configuration.
- Keep host lifecycle and verification behavior in the root `triskel` command; ROS runtime
  code belongs in packages under `ros_ws/src/`.
- Runtime code must use native ROS 2 interfaces and `ros2_control` resource ownership.
