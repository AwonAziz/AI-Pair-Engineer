"""Test JSON response parsing from LLM."""

import sys
from pathlib import Path

# Ensure the project is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_pair_engineer.services.llm import parse_structured_response


def test_parse_raw_json():
    """Parse raw JSON without fences."""
    raw = '{"severity": "high", "title": "Test"}'
    from ai_pair_engineer.models.schemas import Finding
    result = parse_structured_response(raw, Finding)
    assert result.title == "Test"
    assert result.severity.value == "high"


def test_parse_json_in_fences():
    """Parse JSON inside ```json fences."""
    raw = "```json\n{\"severity\": \"high\", \"title\": \"Test\"}\n```"
    from ai_pair_engineer.models.schemas import Finding
    result = parse_structured_response(raw, Finding)
    assert result.title == "Test"


def test_parse_json_with_whitespace():
    """Parse JSON with minor surrounding whitespace/text."""
    raw = "Some prefix text {\"severity\": \"high\", \"title\": \"Test\"} some suffix"
    from ai_pair_engineer.models.schemas import Finding
    result = parse_structured_response(raw, Finding)
    assert result.title == "Test"


def test_parse_malformed_json():
    """Raise ValueError for truly malformed JSON."""
    raw = "This is not JSON at all {{{"
    from ai_pair_engineer.models.schemas import Finding
    try:
        result = parse_structured_response(raw, Finding)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unable to parse" in str(e)