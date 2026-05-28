# Python Execution Rules

- Always use `uv run` to execute Python scripts or tools.
- Example: Use `uv run python app.py` instead of `python app.py`.
- Example: Use `uv run pytest` instead of `pytest`.

# Verification and Code Quality Rules

Before submitting major changes, run the project verification tools to ensure correct style, formatting, types, and that all tests pass. You do not need to run these for minor changes or small tweaks, only when major changes land. These commands are all defined in the pre-commit configuration (`.pre-commit-config.yaml`):

- **Formatting**: Format the Python codebase:
  ```bash
  uv run ruff format
  ```
- **Linting**: Run linting checks and apply automatic fixes:
  ```bash
  uv run ruff check --fix
  ```
- **Type Checking**: Run static type analysis:
  ```bash
  uv run ty check
  ```
- **Testing**: Run the unit test suite:
  ```bash
  uv run pytest
  ```

