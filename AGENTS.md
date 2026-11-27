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

- Keep the root `README.md` lightweight: a concise project description and a link
  to the documentation index. Keep project documentation in `docs/`; do not add
  component READMEs under source, model, or script directories. `AGENTS.md` stays
  at the root for contributor instructions. Store documentation images and
  animations in `docs/assets/` and embed them using relative paths.
- Write for humans. Focus on high-level structure, responsibilities, and common
  workflows. Use short paragraphs and only the examples needed to get started.
  Keep low-level contracts, exhaustive options, and implementation details in
  code, types, docstrings, and CLI help; link to those sources instead of copying them.
- Follow a consistent Markdown format: one sentence-case H1, a short purpose
  paragraph, focused sentence-case H2 sections as needed, and a final `See also`
  section linking to the docs index and relevant guides or source. Use descriptive,
  relative links and fenced code blocks with a language. Commands assume the
  repository root unless stated otherwise.
- Use a single source of truth: give each topic one authoritative home. Link to
  that page from related docs rather than repeating setup steps, explanations,
  or configuration reference.
  Keep `docs/README.md` as the navigation index and group related specialized
  guides in subdirectories.
- Update the relevant documentation in the same change whenever user workflows,
  setup, configuration, interfaces, or architectural responsibilities change.
  Keep updates proportional to the change; internal refactors with no documented
  impact do not need prose churn. Update the index when adding or moving pages,
  fix incoming links, and check relative links and heading anchors before finishing.
- Describe configurable behavior using the parameter name and what it controls.
  Quote its current value only when the instruction or example depends on it.
