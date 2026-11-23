# Python Execution Rules

- Always use `uv run` to execute Python scripts or tools.
- Example: Use `uv run python app.py` instead of `python app.py`.
- Example: Use `uv run pytest` instead of `pytest`.

# Verification and Code Quality Rules

Enforce type safety wherever practical. Give shared data with a known shape an
explicit type, validate external input at the boundary, and pass typed values
between components so static analysis can catch mismatches.

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

# Backwards Compatibility

- Maintain backwards compatibility only when explicitly requested. When changing
  an interface, update its callers and documentation together.

# Documentation Rules

- Describe configurable behavior using the parameter name and what it controls.
  Quote its current value only when the instruction or example depends on it.
