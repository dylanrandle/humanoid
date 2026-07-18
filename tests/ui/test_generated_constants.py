from scripts.generate_js_constants import (
    OUTPUT_PATH,
    render_javascript_constants,
)


def test_javascript_constants_match_python_contract():
    assert OUTPUT_PATH.read_text() == render_javascript_constants(), (
        "Generated UI constants are stale. Run `uv run python scripts/generate_js_constants.py`."
    )
