# Development

Use the repository checks and generators to keep Python, browser code, and shared
message definitions in sync.

## Set up

Complete [Getting started](getting-started.md), then install development dependencies.
Node.js is needed for UI development and the complete verification suite; the
expected major version is in `.node-version`.

```bash
uv sync --group dev
npm ci
```

## Verify changes

Run the complete verification suite before submitting major changes:

```bash
uv run check
```

This runs the same formatting, linting, type-checking, and test suite as CI. The
Pytest suite includes the Node.js UI tests. The check command runs tests in parallel
and reports the slowest test phases. For a serial run or a focused test, use
`uv run pytest` with an optional test path. To run only the UI tests, use `npm test`.

The check command limits each numerical-library and Mesa software-rendering
worker pool to one thread. Pytest and trial processes still run concurrently;
the limits prevent nested thread pools from overwhelming the runner. See the
[verification environment](../src/humanoid/config/check.py) for these settings.
Trial behavior tests use smaller RGB-D images with shadows disabled so concurrent
software renderers can meet the normal capture freshness limit; production camera
settings are unchanged.

## Browser constants

Python enums, payload keys, process mappings, and API routes are the source of truth for
the browser constants. After changing one of those definitions, regenerate the JavaScript
module:

```bash
uv run python scripts/generate_js_constants.py
```

The Python and UI test suites both verify that the generated module is current.

## LCM types

Regenerate the Python bindings after editing a `.lcm` definition in `src/humanoid/types/lcm/`.

```bash
./scripts/generate-lcm-types
```

## Documentation changes

Update the page that owns an affected workflow or concept in the same change.
Link to existing explanations and source definitions instead of duplicating them.
The [documentation rules](../AGENTS.md#documentation-rules) define the format and scope.

## See also

- [Documentation index](README.md)
- [Architecture and configuration](architecture.md)
- [Verification entry point](../src/humanoid/check.py)
