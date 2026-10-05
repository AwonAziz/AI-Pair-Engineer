"""Tests for structured-response parsing.

Models emit several shapes in practice, and a parser that only handles the
clean one turns a formatting quirk into a pipeline failure.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from ai_pair_engineer.models.schemas import Finding, Severity
from ai_pair_engineer.services.llm import (
    LLMResponseError,
    _extract_balanced_object,
    _extract_fenced_block,
    parse_structured_response,
)


def _finding_json(**overrides: object) -> str:
    """Build a minimal valid Finding payload, overriding individual fields."""
    payload: dict[str, object] = {
        "severity": "high",
        "category": "security",
        "title": "SQL injection",
        "description": "d",
        "recommendation": "r",
    }
    payload.update(overrides)
    return json.dumps(payload)


# A payload with every required Finding field, in its valid form.
VALID = _finding_json()


class TestAcceptedShapes:
    def test_parses_bare_json(self) -> None:
        assert parse_structured_response(VALID, Finding).title == "SQL injection"

    def test_parses_json_inside_a_json_fence(self) -> None:
        raw = f"```json\n{VALID}\n```"
        assert parse_structured_response(raw, Finding).severity is Severity.HIGH

    def test_parses_json_inside_an_unlabelled_fence(self) -> None:
        raw = f"```\n{VALID}\n```"
        assert parse_structured_response(raw, Finding).title == "SQL injection"

    def test_parses_json_surrounded_by_prose(self) -> None:
        raw = f"Here is my analysis:\n{VALID}\nLet me know if you need more."
        assert parse_structured_response(raw, Finding).title == "SQL injection"

    def test_prefers_the_fence_over_surrounding_prose(self) -> None:
        raw = f'I considered `{{"title": "wrong"}}` first.\n```json\n{VALID}\n```'
        assert parse_structured_response(raw, Finding).title == "SQL injection"

    def test_tolerates_whitespace_around_the_object(self) -> None:
        assert parse_structured_response(f"\n\n  {VALID}  \n\n", Finding).severity is Severity.HIGH

    def test_handles_a_truncated_fence(self) -> None:
        """A response cut off by max_tokens still has usable content."""
        raw = "```json\n" + VALID
        assert parse_structured_response(raw, Finding).title == "SQL injection"

    def test_handles_braces_inside_string_values(self) -> None:
        """Refactored code inside a JSON field contains unbalanced braces."""
        raw = json.dumps({"title": "t", "body": "def f(): return {1: 2}", "note": "has } inside"})
        assert "return {1: 2}" in parse_structured_response(raw, _Loose).body

    def test_handles_escaped_quotes_inside_values(self) -> None:
        raw = json.dumps({"title": 'say "hi"', "note": 'a " quote'})
        assert parse_structured_response(raw, _Loose).title == 'say "hi"'


class TestRejectedShapes:
    def test_raises_when_there_is_no_json(self) -> None:
        with pytest.raises(LLMResponseError, match="could not recover"):
            parse_structured_response("This is prose only.", Finding)

    def test_raises_on_truncated_json(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_structured_response('{"severity": "high", "title":', Finding)

    def test_raises_on_unbalanced_braces(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_structured_response('{"severity": "high" "title": "x"}', Finding)

    def test_raises_when_required_fields_are_missing(self) -> None:
        with pytest.raises(LLMResponseError, match="schema validation failed"):
            parse_structured_response('{"title": "only a title"}', Finding)

    def test_raises_on_an_enum_outside_the_allowed_set(self) -> None:
        raw = _finding_json(severity="apocalyptic")
        with pytest.raises(LLMResponseError, match="severity"):
            parse_structured_response(raw, Finding)

    def test_raises_when_a_float_is_out_of_range(self) -> None:
        raw = _finding_json(severity="low", confidence=7)
        with pytest.raises(LLMResponseError, match="confidence"):
            parse_structured_response(raw, Finding)

    def test_raises_on_a_json_array(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_structured_response('[{"title": "t"}]', Finding)

    def test_reports_the_offending_field_in_the_message(self) -> None:
        raw = _finding_json(severity="low", confidence=7)
        with pytest.raises(LLMResponseError, match="confidence"):
            parse_structured_response(raw, Finding)


class TestBalancedObjectExtraction:
    def test_finds_a_simple_object(self) -> None:
        assert _extract_balanced_object('prose {"a": 1} more') == '{"a": 1}'

    def test_returns_none_without_any_object(self) -> None:
        assert _extract_balanced_object("no braces here") is None

    def test_handles_nested_objects(self) -> None:
        raw = '{"a": {"b": {"c": 1}}}'
        assert _extract_balanced_object(raw) == raw

    def test_ignores_braces_inside_strings(self) -> None:
        raw = '{"a": "}{", "b": 1}'
        assert _extract_balanced_object(raw) == raw

    def test_returns_none_for_an_unterminated_object(self) -> None:
        assert _extract_balanced_object('{"a": 1') is None


class TestFenceExtraction:
    def test_extracts_a_closed_fence(self) -> None:
        assert _extract_fenced_block("```json\n{}\n```") == "{}"

    def test_extracts_an_unterminated_fence(self) -> None:
        assert _extract_fenced_block("```json\n{}") == "{}"

    def test_stops_at_the_first_closing_fence(self) -> None:
        raw = '```json\n{"a": 1}\n```\n```python\nprint(1)\n```'
        assert _extract_fenced_block(raw) == '{"a": 1}'

    def test_returns_none_without_a_fence(self) -> None:
        assert _extract_fenced_block('{"a": 1}') is None


class _Loose(BaseModel):
    """Permissive model for cases where the exact schema is not under test."""

    title: str = ""
    body: str = ""
    note: str = ""
