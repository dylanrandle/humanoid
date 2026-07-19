# Python Execution Rules

- Always use `uv run` to execute Python scripts or tools.
- Example: Use `uv run python app.py` instead of `python app.py`.
- Example: Use `uv run pytest` instead of `pytest`.

# Verification and Code Quality Rules

Before submitting major changes, run the complete project verification suite. You do not need to run it for minor changes or small tweaks:

```bash
uv run check
```

The command runs formatting checks, linting, static type analysis, and the complete test suite. To apply automatic formatting or lint fixes while working, use:

```bash
uv run ruff format
uv run ruff check --fix
```

# Project Organization Rules

- Centralize shared type definitions in `src/humanoid/types/` and import them from there.
- Keep concrete configuration instances, including project defaults, in
  `src/humanoid/config/`; modules in `src/humanoid/types/` should define configuration
  shapes without instantiating them.
