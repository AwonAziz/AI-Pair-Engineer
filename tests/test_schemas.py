"""Tests for the real Pydantic schemas.

The original test file re-declared local stand-in classes for every model, so
it passed while asserting nothing about the shipped code. These tests import the
actual schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_pair_engineer.models.schemas import (
    REFACTOR_CATEGORIES,
    SEVERITY_WEIGHTS,
    TEST_CATEGORIES,
    AnalysisResult,
    Category,
    FinalReview,
    Finding,
    Location,
    RefactorChange,
    RefactorResult,
    Severity,
    TestCase,
    TestResult,
)


def _finding(**overrides: object) -> Finding:
    payload: dict[str, object] = {
        "severity": Severity.MEDIUM,
        "category": Category.MAINTAINABILITY,
        "title": "Long function",
        "description": "Does too much.",
        "recommendation": "Split it.",
        "confidence": 0.9,
    }
    payload.update(overrides)
    return Finding(**payload)  # type: ignore[arg-type]


class TestFinding:
    def test_accepts_a_well_formed_finding(self) -> None:
        finding = _finding()
        assert finding.severity is Severity.MEDIUM
        assert finding.weight == SEVERITY_WEIGHTS[Severity.MEDIUM]

    def test_confidence_defaults_to_full(self) -> None:
        finding = _finding(confidence=1.0)
        assert finding.confidence == 1.0

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_rejects_out_of_range_confidence(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="confidence"):
            _finding(confidence=bad)

    @pytest.mark.parametrize("field", ["title", "description", "recommendation"])
    def test_rejects_blank_text(self, field: str) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            _finding(**{field: "   "})

    def test_rejects_unknown_severity(self) -> None:
        with pytest.raises(ValidationError):
            _finding(severity="catastrophic")

    def test_rejects_unknown_category(self) -> None:
        with pytest.raises(ValidationError):
            _finding(category="vibes")

    def test_weight_matches_the_severity_table(self) -> None:
        for severity, weight in SEVERITY_WEIGHTS.items():
            assert _finding(severity=severity).weight == weight


class TestLocation:
    def test_renders_single_line(self) -> None:
        assert Location(line=12, symbol="handler").render() == "line 12, in `handler`"

    def test_renders_a_range_when_the_ends_differ(self) -> None:
        location = Location(line=3, end_line=9)
        assert location.render() == "lines 3-9"

    def test_renders_nothing_when_absent(self) -> None:
        assert Location().render() == ""

    @pytest.mark.parametrize("bad", [0, -5])
    def test_rejects_line_numbers_below_one(self, bad: int) -> None:
        with pytest.raises(ValidationError):
            Location(line=bad)


class TestAnalysisResult:
    def test_counts_severities_including_zeroes(self) -> None:
        result = AnalysisResult(summary="s", findings=[_finding(), _finding(severity="low")])
        counts = result.severity_counts()
        assert counts[Severity.MEDIUM] == 1
        assert counts[Severity.LOW] == 1
        assert counts[Severity.CRITICAL] == 0

    def test_quality_score_starts_at_one_hundred(self) -> None:
        assert AnalysisResult(summary="clean", findings=[]).quality_score() == 100

    def test_quality_score_deducts_by_severity_weight(self) -> None:
        findings = [_finding(severity=Severity.HIGH)]
        expected = 100 - SEVERITY_WEIGHTS[Severity.HIGH]
        assert AnalysisResult(summary="s", findings=findings).quality_score() == expected

    def test_quality_score_floors_at_zero(self) -> None:
        findings = [_finding(severity=Severity.CRITICAL) for _ in range(10)]
        assert AnalysisResult(summary="s", findings=findings).quality_score() == 0

    def test_blocking_includes_only_critical_and_high(self) -> None:
        result = AnalysisResult(
            summary="s",
            findings=[
                _finding(severity=Severity.LOW),
                _finding(severity=Severity.MEDIUM),
                _finding(severity=Severity.HIGH),
                _finding(severity=Severity.CRITICAL),
            ],
        )
        assert {f.severity for f in result.blocking()} == {Severity.HIGH, Severity.CRITICAL}

    def test_findings_default_to_empty(self) -> None:
        assert AnalysisResult(summary="s").findings == []


class TestTestSchemas:
    def test_parses_a_test_case(self) -> None:
        case = TestCase(name="test_x", purpose="p", test_code="assert True", covers=["a"])
        assert case.covers == ["a"]

    def test_rejects_blank_test_code(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            TestCase(name="n", purpose="p", test_code="  ")

    def test_result_defaults_to_no_tests(self) -> None:
        assert TestResult().tests == []


class TestRefactorSchemas:
    def test_parses_a_change(self) -> None:
        change = RefactorChange(title="t", description="d", rationale="r", addresses=["f"])
        assert change.addresses == ["f"]

    def test_rejects_blank_refactored_code(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            RefactorResult(summary="s", refactored_code="\n\n")

    def test_optional_collections_default_empty(self) -> None:
        result = RefactorResult(summary="s", refactored_code="pass")
        assert result.changes == [] and result.risks == []


class TestFinalReview:
    def test_accepts_a_full_review(self) -> None:
        review = FinalReview(
            approved=True,
            score=90,
            verdict="approve",
            summary="clean",
            strengths=["clearer"],
            remaining_issues=[],
            regression_risk="low",
            recommendation="merge",
        )
        assert review.verdict == "approve"
        assert review.regression_risk == "low"

    @pytest.mark.parametrize("score", [-1, 101])
    def test_rejects_scores_outside_zero_to_one_hundred(self, score: int) -> None:
        with pytest.raises(ValidationError, match="score"):
            FinalReview(approved=False, score=score, recommendation="r")

    def test_rejects_unknown_verdict(self) -> None:
        with pytest.raises(ValidationError):
            FinalReview(approved=False, score=50, verdict="shipit", recommendation="r")

    def test_rejects_unknown_regression_risk(self) -> None:
        with pytest.raises(ValidationError):
            FinalReview(
                approved=False,
                score=50,
                regression_risk="spicy",
                recommendation="r",
            )

    def test_defaults_are_conservative(self) -> None:
        review = FinalReview(approved=False, score=50, recommendation="r")
        assert review.verdict == "needs_review"
        assert review.regression_risk == "medium"


class TestCategorySets:
    def test_refactor_categories_exclude_failure_categories(self) -> None:
        assert Category.SECURITY not in REFACTOR_CATEGORIES
        assert Category.ERROR_HANDLING not in REFACTOR_CATEGORIES

    def test_refactor_categories_include_structure_categories(self) -> None:
        assert Category.COMPLEXITY in REFACTOR_CATEGORIES
        assert Category.DUPLICATION in REFACTOR_CATEGORIES

    def test_test_categories_are_the_failure_paths(self) -> None:
        assert {Category.SECURITY, Category.ERROR_HANDLING} == TEST_CATEGORIES
