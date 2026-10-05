"""Tests for finding-to-defect matching.

Matching is where a benchmark decides whether it measures anything. A loose
match inflates recall; a strict one flatters precision. These tests pin the
behaviour in both directions, including the specific ways a match must *not*
fire.
"""

from __future__ import annotations

import pytest
from benchmarks.matching import (
    finding_text,
    keyword_match,
    match_findings,
    match_rule_for,
    matched_defect_ids,
)
from benchmarks.models import MatchRule, PlantedDefect
from pydantic import ValidationError

from ai_pair_engineer.models.schemas import Category, Finding, Location, Severity


def _defect(**overrides: object) -> PlantedDefect:
    payload: dict[str, object] = {
        "id": "sqli",
        "summary": "Query built by interpolation",
        "category": Category.SECURITY,
        "severity": Severity.CRITICAL,
        "line": 14,
        "keywords": (("sql", "query"), ("interpolat", "f-string", "concat")),
    }
    payload.update(overrides)
    return PlantedDefect(**payload)  # type: ignore[arg-type]


def _finding(title: str = "SQL injection", line: int | None = None, **overrides: object) -> Finding:
    payload: dict[str, object] = {
        "severity": Severity.CRITICAL,
        "category": Category.SECURITY,
        "title": title,
        "description": "The query is built by string interpolation.",
        "recommendation": "Use a parameterised query.",
        "confidence": 0.9,
    }
    if line is not None:
        payload["location"] = Location(line=line)
    payload.update(overrides)
    return Finding(**payload)  # type: ignore[arg-type]


class TestKeywordMatch:
    def test_matches_when_every_group_is_present(self) -> None:
        defect = _defect()
        text = "the query uses f-string interpolation"
        assert keyword_match(defect, text)

    def test_matches_when_alternatives_are_used(self) -> None:
        assert keyword_match(_defect(), "query built via concatenation")

    def test_fails_when_a_group_is_absent(self) -> None:
        assert not keyword_match(_defect(), "a parameterised query is missing")

    def test_fails_on_an_unrelated_finding(self) -> None:
        assert not keyword_match(_defect(), "the function is a little long")

    def test_a_defect_without_keywords_never_matches_on_text(self) -> None:
        """Otherwise any finding at all could be credited to it."""
        assert not keyword_match(_defect(keywords=()), "sql interpolation query")

    def test_is_case_insensitive(self) -> None:
        assert keyword_match(_defect(), "SQL QUERY WITH INTERPOLATION")

    def test_all_groups_must_match_not_any(self) -> None:
        """The point of groups: a single shared word is not evidence."""
        assert not keyword_match(_defect(), "this function has a slow query")


class TestFindingText:
    def test_includes_title_description_and_recommendation(self) -> None:
        text = finding_text(_finding())
        assert "sql injection" in text
        assert "interpolation" in text
        assert "parameterised" in text

    def test_includes_the_enclosing_symbol(self) -> None:
        finding = _finding(location=Location(line=3, symbol="lookup_user"))
        assert "lookup_user" in finding_text(finding)

    def test_handles_a_finding_with_no_location(self) -> None:
        assert "sql injection" in finding_text(_finding())


class TestMatchRule:
    def test_both_rules_fire_for_a_line_and_text_match(self) -> None:
        assert match_rule_for(_finding(line=14), _defect()) is MatchRule.BOTH

    def test_located_alone_is_enough(self) -> None:
        """A model that reports the right line has found the right thing."""
        finding = _finding(title="Unclear issue", line=14, description="vague")
        assert match_rule_for(finding, _defect()) is MatchRule.LOCATED

    def test_semantic_alone_is_enough(self) -> None:
        """A model that miscounts lines should still get credit."""
        finding = _finding(line=99)
        assert match_rule_for(finding, _defect()) is MatchRule.SEMANTIC

    def test_no_rule_fires_for_an_unrelated_finding(self) -> None:
        finding = _finding(title="Unused import", line=99, description="os is unused")
        assert match_rule_for(finding, _defect()) is None

    def test_a_finding_with_no_line_can_still_match_on_text(self) -> None:
        assert match_rule_for(_finding(), _defect()) is MatchRule.SEMANTIC

    def test_adjacent_lines_are_within_tolerance(self) -> None:
        finding = _finding(title="x", line=15, description="vague", recommendation="vague")
        assert match_rule_for(finding, _defect()) is MatchRule.LOCATED

    def test_distant_lines_do_not_match_by_location(self) -> None:
        finding = _finding(title="x", line=40, description="nothing relevant")
        assert match_rule_for(finding, _defect()) is None

    def test_a_line_span_matches_any_line_inside_it(self) -> None:
        defect = _defect(end_line=20, keywords=())
        assert match_rule_for(_finding(title="x", line=18), defect) is MatchRule.LOCATED
        assert match_rule_for(_finding(title="x", line=25), defect) is None

    def test_a_finding_with_no_line_cannot_match_by_location(self) -> None:
        """A model reporting no location must be credited on text or not at all."""
        finding = _finding(title="x", description="vague", recommendation="vague")
        assert match_rule_for(finding, _defect()) is None

    def test_the_schema_rejects_line_zero(self) -> None:
        """Line 0 is not a real line, so Location refuses to carry it.

        A model that emits 0 is signalling it had no idea; silently clamping it
        to line 1 would credit it against whatever sits there.
        """
        with pytest.raises(ValidationError):
            Location(line=0)

    def test_the_symbol_is_searched_but_a_wrong_symbol_does_not_block(self) -> None:
        finding = _finding(location=Location(line=None, symbol="somewhere_else"))
        assert match_rule_for(finding, _defect()) is MatchRule.SEMANTIC


class TestMatchFindings:
    def test_records_one_match_per_detectable_finding(self) -> None:
        matches, unmatched = match_findings([_finding(line=14)], (_defect(),))
        assert len(matches) == 1
        assert unmatched == []

    def test_unrelated_findings_are_returned_as_unmatched(self) -> None:
        noise = _finding(title="Unused import os", line=99, description="never referenced")
        matches, unmatched = match_findings([noise], (_defect(),))
        assert matches == []
        assert unmatched == [noise]

    def test_two_findings_for_one_defect_are_both_credited(self) -> None:
        """Duplicate reports stay visible instead of being collapsed away."""
        matches, _ = match_findings(
            [_finding(line=14), _finding(line=14, title="Second report")],
            (_defect(),),
        )
        assert len(matches) == 2
        assert matched_defect_ids(matches) == {"sqli"}

    def test_a_finding_matches_at_most_one_defect(self) -> None:
        """Otherwise a vague finding could credit several defects at once."""
        first = _defect(id="a")
        second = _defect(id="b", line=14, keywords=(("query",), ("interpolat",)))
        matches, _ = match_findings([_finding(line=14)], (first, second))
        assert len(matches) == 1

    def test_empty_findings_produce_nothing(self) -> None:
        assert match_findings([], (_defect(),)) == ([], [])

    def test_no_defects_leaves_everything_unmatched(self) -> None:
        finding = _finding()
        matches, unmatched = match_findings([finding], ())
        assert matches == []
        assert unmatched == [finding]

    def test_empty_corpus_defect_list_is_safe(self) -> None:
        matches, unmatched = match_findings([_finding()], ())
        assert not matches and unmatched


class TestMatchedDefectIds:
    def test_deduplicates_across_findings(self) -> None:
        matches, _ = match_findings([_finding(line=14), _finding(line=14)], (_defect(),))
        assert matched_defect_ids(matches) == {"sqli"}

    def test_returns_empty_for_no_matches(self) -> None:
        assert matched_defect_ids([]) == set()


class TestUnrelatedDefectIsolation:
    def test_a_readability_defect_does_not_credit_a_security_finding(self) -> None:
        """The failure mode that made the old reviewer filter worth fixing."""
        readability = _defect(
            id="naming",
            category=Category.READABILITY,
            severity=Severity.LOW,
            keywords=(("naming", "name"), ("inconsistent", "style")),
        )
        matches, _ = match_findings([_finding(line=None)], (readability,))
        assert matches == []


@pytest.mark.parametrize("tolerance_line", [13, 14, 15])
def test_tolerance_window_is_one_line_either_side(tolerance_line: int) -> None:
    finding = _finding(title="x", line=tolerance_line, description="vague", recommendation="v")
    assert match_rule_for(finding, _defect()) is MatchRule.LOCATED
