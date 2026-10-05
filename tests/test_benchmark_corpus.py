"""Tests for corpus loading, validation, and scoring.

The corpus is ground truth, so these tests are as much about protecting the
benchmark from itself as about testing the loader. A corpus whose annotations
have drifted would report a confident number that means nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.corpus import (
    CorpusError,
    load_case,
    load_cases,
    load_regressions,
    validate_case,
)
from benchmarks.models import (
    CaseResult,
    DetectionCase,
    Match,
    MatchRule,
    PlantedDefect,
    Tier,
)
from benchmarks.scoring import (
    aggregate,
    recall_by_case,
    recall_by_category,
    recall_by_severity,
    score_case,
    severity_weighted_recall,
)
from benchmarks.validate import validate_corpus

from ai_pair_engineer.models.schemas import Category, Severity


class TestTheShippedCorpusIsValid:
    """The corpus on disk must be consistent, or every number is suspect."""

    def test_reports_no_problems(self) -> None:
        problems, _, _ = validate_corpus()
        assert problems == [], "\n".join(problems)

    def test_contains_more_than_one_case(self) -> None:
        _, case_count, _ = validate_corpus()
        assert case_count >= 5

    def test_contains_planted_defects(self) -> None:
        _, _, defect_count = validate_corpus()
        assert defect_count >= 15

    def test_contains_at_least_one_clean_case(self) -> None:
        """Without a clean case, false positives cannot be measured."""
        assert any(case.clean for case in load_cases())

    def test_every_case_parses(self) -> None:
        """A case that does not parse is not a real review target."""
        from ai_pair_engineer.static import analyze_source

        for case in load_cases():
            report = analyze_source(case.source, case.language)
            assert report.syntax_error is None, f"{case.id}: {report.syntax_error}"

    def test_every_case_compiles(self) -> None:
        for case in load_cases():
            compile(case.source, f"{case.id}.py", "exec")

    def test_ids_are_unique(self) -> None:
        ids = [case.id for case in load_cases()]
        assert len(ids) == len(set(ids))

    def test_defect_ids_are_unique_within_each_case(self) -> None:
        for case in load_cases():
            ids = [defect.id for defect in case.defects]
            assert len(ids) == len(set(ids)), case.id

    def test_clean_cases_declare_no_defects(self) -> None:
        for case in load_cases():
            if case.clean:
                assert case.defects == ()

    def test_every_case_documents_its_intent(self) -> None:
        for case in load_cases():
            assert case.intent.strip(), case.id


class TestLineNumbersPointAtRealCode:
    """A line number that drifted points the benchmark at the wrong code."""

    def test_every_declared_line_exists(self) -> None:
        for case in load_cases():
            for defect in case.defects:
                assert 1 <= defect.line <= len(case.source.splitlines()), (
                    f"{case.id}/{defect.id} declares line {defect.line}"
                )

    def test_no_defect_declares_a_line_beyond_the_source(self) -> None:
        for case in load_cases():
            line_count = len(case.source.splitlines())
            for defect in case.defects:
                assert defect.line <= line_count

    def test_spans_are_well_formed(self) -> None:
        for case in load_cases():
            for defect in case.defects:
                if defect.end_line is not None:
                    assert defect.line <= defect.end_line

    def test_every_defect_has_enough_keyword_groups(self) -> None:
        """One group matches too easily to be evidence."""
        for case in load_cases():
            for defect in case.defects:
                assert len(defect.keywords) >= 2, f"{case.id}/{defect.id}"

    def test_every_defect_has_a_summary_and_note(self) -> None:
        for case in load_cases():
            for defect in case.defects:
                assert defect.summary.strip(), f"{case.id}/{defect.id}"
                assert defect.note.strip(), f"{case.id}/{defect.id}"


class TestRegressionCorpus:
    def test_loads_several_cases(self) -> None:
        assert len(load_regressions()) >= 3

    def test_pairs_differ_in_real_code(self) -> None:
        """A pair differing only in comments has no behaviour change to catch."""
        for case in load_regressions():
            original = [
                line.strip()
                for line in case.original.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            refactored = [
                line.strip()
                for line in case.refactored.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            assert original != refactored, case.id

    def test_every_case_describes_the_change(self) -> None:
        for case in load_regressions():
            assert case.change.strip(), case.id

    def test_every_case_has_expected_phrases(self) -> None:
        for case in load_regressions():
            assert case.expected, case.id

    def test_expected_phrases_are_specific_enough_to_mean_something(self) -> None:
        """A vague phrase would credit a reviewer that noticed nothing."""
        from benchmarks.validate import GENERIC_PHRASES

        for case in load_regressions():
            for phrase in case.expected:
                assert phrase.lower() not in GENERIC_PHRASES, f"{case.id}: {phrase}"

    def test_both_versions_compile(self) -> None:
        for case in load_regressions():
            compile(case.original, f"{case.id}-original.py", "exec")
            compile(case.refactored, f"{case.id}-refactored.py", "exec")

    def test_ids_are_unique(self) -> None:
        ids = [case.id for case in load_regressions()]
        assert len(ids) == len(set(ids))

    def test_covers_a_range_of_change_kinds(self) -> None:
        """A corpus of one failure mode would flatter one prompt."""
        tags = {tag for case in load_regressions() for tag in case.tags}
        assert len(tags) >= 4


class TestLoaderErrors:
    def _write_case(self, tmp_path: Path, manifest: str, source: str = "x = 1\n") -> Path:
        directory = tmp_path / "c"
        directory.mkdir()
        (directory / "case.toml").write_text(manifest, encoding="utf-8")
        (directory / "source.py").write_text(source, encoding="utf-8")
        return directory

    def test_reports_a_missing_manifest(self, tmp_path: Path) -> None:
        directory = tmp_path / "c"
        directory.mkdir()
        (directory / "source.py").write_text("x = 1\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="missing manifest"):
            load_case(directory)

    def test_reports_a_missing_source(self, tmp_path: Path) -> None:
        directory = tmp_path / "c"
        directory.mkdir()
        (directory / "case.toml").write_text('id = "c"\n', encoding="utf-8")
        with pytest.raises(CorpusError, match=r"no source\.py"):
            load_case(directory)

    def test_reports_invalid_toml(self, tmp_path: Path) -> None:
        directory = self._write_case(tmp_path, "this is not = = toml")
        with pytest.raises(CorpusError, match="invalid TOML"):
            load_case(directory)

    def test_reports_a_missing_required_field(self, tmp_path: Path) -> None:
        directory = self._write_case(
            tmp_path,
            'id = "c"\n[[defects]]\nid = "d"\nsummary = "s"\ncategory = "security"\n',
        )
        with pytest.raises(CorpusError, match="severity"):
            load_case(directory)

    def test_reports_an_unknown_category(self, tmp_path: Path) -> None:
        directory = self._write_case(
            tmp_path,
            'id = "c"\n'
            '[[defects]]\nid = "d"\nsummary = "s"\ncategory = "vibes"\n'
            'severity = "high"\nline = 1\n',
        )
        with pytest.raises(CorpusError, match="unknown category"):
            load_case(directory)

    def test_reports_an_unknown_severity(self, tmp_path: Path) -> None:
        directory = self._write_case(
            tmp_path,
            'id = "c"\n'
            '[[defects]]\nid = "d"\nsummary = "s"\ncategory = "security"\n'
            'severity = "apocalyptic"\nline = 1\n',
        )
        with pytest.raises(CorpusError, match="unknown severity"):
            load_case(directory)

    def test_rejects_a_clean_case_with_defects(self, tmp_path: Path) -> None:
        directory = self._write_case(
            tmp_path,
            'id = "c"\nclean = true\n'
            '[[defects]]\nid = "d"\nsummary = "s"\ncategory = "security"\n'
            'severity = "high"\nline = 1\n',
        )
        with pytest.raises(CorpusError, match="marked clean"):
            load_case(directory)

    def test_rejects_a_regression_with_no_expected_phrases(self, tmp_path: Path) -> None:
        directory = tmp_path / "r"
        directory.mkdir()
        (directory / "case.toml").write_text('id = "r"\n', encoding="utf-8")
        (directory / "original.py").write_text("x = 1\n", encoding="utf-8")
        (directory / "refactored.py").write_text("x = 2\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="expected"):
            load_regressions(tmp_path)

    def test_rejects_a_regression_missing_a_version_file(self, tmp_path: Path) -> None:
        directory = tmp_path / "r"
        directory.mkdir()
        (directory / "case.toml").write_text('id = "r"\nexpected = ["x"]\n', encoding="utf-8")
        (directory / "original.py").write_text("x = 1\n", encoding="utf-8")
        with pytest.raises(CorpusError, match=r"refactored\.py"):
            load_regressions(tmp_path)

    def test_a_missing_regression_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert load_regressions(tmp_path / "absent") == []


class TestValidateCase:
    def test_reports_a_line_past_the_end(self) -> None:
        case = DetectionCase(
            id="x",
            language="python",
            source="x = 1\n",
            defects=(_defect(line=99),),
        )
        problems = validate_case(case)
        assert any("outside" in problem for problem in problems)

    def test_reports_a_defect_with_no_keywords(self) -> None:
        case = DetectionCase(
            id="x", language="python", source="x = 1\n", defects=(_defect(line=1, keywords=()),)
        )
        assert any("no keywords" in problem for problem in validate_case(case))

    def test_reports_an_empty_source(self) -> None:
        empty = DetectionCase(id="x", language="python", source="")
        assert any("empty" in p for p in validate_case(empty))

    def test_returns_nothing_for_a_good_case(self) -> None:
        case = DetectionCase(
            id="x", language="python", source="x = 1\ny = 2\n", defects=(_defect(line=1),)
        )
        assert validate_case(case) == []


class TestScoreCase:
    def test_records_missed_defects(self) -> None:
        case = DetectionCase(
            id="x",
            language="python",
            source="x = 1\n",
            defects=(_defect(id="a", line=1), _defect(id="b", line=2)),
        )
        result = score_case(
            case,
            CaseResult(case_id="x", tier=Tier.STATIC, matches=[_match("a")]),
        )
        assert result.missed == ["b"]

    def test_records_nothing_missed_when_all_are_detected(self) -> None:
        case = DetectionCase(
            id="x", language="python", source="x = 1\n", defects=(_defect(id="a", line=1),)
        )
        result = score_case(
            case,
            CaseResult(case_id="x", tier=Tier.STATIC, matches=[_match("a")]),
        )
        assert result.missed == []

    def test_ignores_unknown_defect_ids(self) -> None:
        case = DetectionCase(id="x", language="python", source="x = 1\n", defects=())
        result = score_case(
            case,
            CaseResult(case_id="x", tier=Tier.STATIC, matches=[_match("ghost")]),
        )
        assert result.missed == []


def _match(defect_id: str) -> Match:
    return Match(defect_id=defect_id, finding_title="t", rule=MatchRule.LOCATED)


def _defect(
    *,
    id: str = "d",
    line: int = 1,
    category: Category = Category.SECURITY,
    severity: Severity = Severity.HIGH,
    keywords: tuple[tuple[str, ...], ...] = (("a",), ("b",)),
) -> object:

    return PlantedDefect(
        id=id,
        summary="summary",
        category=category,
        severity=severity,
        line=line,
        keywords=keywords,
        note="note",
    )


def _case(
    id: str,
    defects: tuple[object, ...] = (),
    clean: bool = False,
) -> DetectionCase:
    return DetectionCase(
        id=id,
        language="python",
        source="x = 1\n",
        defects=defects,  # type: ignore[arg-type]
        clean=clean,
    )


def _result(
    case_id: str,
    *,
    detected: tuple[str, ...] = (),
    findings: int = 0,
    error: str | None = None,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        tier=Tier.STATIC,
        findings=[object()] * findings,  # type: ignore[list-item]
        matches=[Match(d, "title", MatchRule.LOCATED) for d in detected],
        error=error,
    )


class TestAggregate:
    def test_counts_planted_and_detected(self) -> None:
        pairs = [(_case("a", (_defect(id="x"), _defect(id="y"))), _result("a", detected=("x",)))]
        score = aggregate(Tier.STATIC, pairs)
        assert score.planted == 2
        assert score.detected == 1

    def test_recall_is_detected_over_planted(self) -> None:
        pairs = [(_case("a", (_defect(id="x"), _defect(id="y"))), _result("a", detected=("x",)))]
        assert aggregate(Tier.STATIC, pairs).recall == pytest.approx(0.5)

    def test_recall_is_none_with_no_planted_defects(self) -> None:
        assert aggregate(Tier.STATIC, [(_case("a", clean=True), _result("a"))]).recall is None

    def test_counts_clean_case_findings_as_noise(self) -> None:
        pairs = [(_case("clean", clean=True), _result("clean", findings=3))]
        score = aggregate(Tier.STATIC, pairs)
        assert score.clean_case_findings == 3
        assert score.clean_noise == pytest.approx(3.0)

    def test_clean_noise_is_none_with_no_clean_cases(self) -> None:
        pairs = [(_case("a", (_defect(),)), _result("a", detected=("d",)))]
        assert aggregate(Tier.STATIC, pairs).clean_noise is None

    def test_groundedness_is_matched_over_findings(self) -> None:
        pairs = [(_case("a", (_defect(),)), _result("a", detected=("d",), findings=4))]
        assert aggregate(Tier.STATIC, pairs).groundedness == pytest.approx(0.25)

    def test_a_failed_case_contributes_no_counts(self) -> None:
        pairs = [(_case("a", (_defect(),)), _result("a", error="boom"))]
        score = aggregate(Tier.STATIC, pairs)
        assert score.planted == 0
        assert score.cases == 1

    def test_groundedness_is_none_with_no_findings(self) -> None:
        assert aggregate(Tier.STATIC, [(_case("a", clean=True), _result("a"))]).groundedness is None


class TestRecallBreakdowns:
    def _pairs(self) -> list[tuple[DetectionCase, CaseResult]]:
        return [
            (
                _case(
                    "a",
                    (
                        _defect(id="s", category=Category.SECURITY, severity=Severity.CRITICAL),
                        _defect(id="c", category=Category.COMPLEXITY, severity=Severity.LOW),
                    ),
                ),
                _result("a", detected=("s",)),
            )
        ]

    def test_macro_recall_averages_per_case(self) -> None:
        assert recall_by_case(self._pairs()) == pytest.approx(0.5)

    def test_macro_recall_ignores_cases_without_defects(self) -> None:
        pairs = [(_case("clean", clean=True), _result("clean"))]
        assert recall_by_case(pairs) is None

    def test_macro_recall_ignores_failed_cases(self) -> None:
        pairs = [(_case("a", (_defect(),)), _result("a", error="boom"))]
        assert recall_by_case(pairs) is None

    def test_category_recall_reports_one(self) -> None:
        breakdown = recall_by_category(self._pairs())
        assert breakdown[Category.SECURITY] == pytest.approx(1.0)
        assert breakdown[Category.COMPLEXITY] == pytest.approx(0.0)

    def test_category_recall_omits_uncategorised(self) -> None:
        assert Category.PERFORMANCE not in recall_by_category(self._pairs())

    def test_severity_recall_reports_each_severity(self) -> None:
        breakdown = recall_by_severity(self._pairs())
        assert breakdown[Severity.CRITICAL] == pytest.approx(1.0)
        assert breakdown[Severity.LOW] == pytest.approx(0.0)

    def test_severity_recall_orders_most_serious_first(self) -> None:
        breakdown = recall_by_severity(self._pairs())
        assert next(iter(breakdown)) is Severity.CRITICAL

    def test_severity_weighted_recall_weights_critical_above_low(self) -> None:
        """Missing a critical must cost more than missing a low."""
        pairs = [
            (
                _case(
                    "a",
                    (
                        _defect(id="crit", severity=Severity.CRITICAL),
                        _defect(id="low", severity=Severity.LOW),
                    ),
                ),
                _result("a", detected=("crit",)),
            )
        ]
        # One critical caught of two weights, one low missed of one weight.
        assert severity_weighted_recall(pairs) == pytest.approx(4 / 5)

    def test_severity_weighted_recall_is_none_with_nothing_planted(self) -> None:
        pairs = [(_case("clean", clean=True), _result("clean"))]
        assert severity_weighted_recall(pairs) is None
