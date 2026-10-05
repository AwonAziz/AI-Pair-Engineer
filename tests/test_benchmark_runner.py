"""Tests for the benchmark runner, cache, and report rendering.

The runner is where money gets spent, so the failure modes worth pinning are:
a cached result must not be reused after the code under review changes, a
crashing case must not end the run, and the report must not overstate what the
numbers mean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from benchmarks import runner as runner_module
from benchmarks.models import (
    CaseResult,
    DetectionCase,
    Match,
    MatchRule,
    PlantedDefect,
    RegressionResult,
    RunReport,
    Tier,
    TierScore,
)
from benchmarks.report import render_json, render_markdown, tier_order
from benchmarks.runner import review_with_static, run_benchmark

from ai_pair_engineer.models.schemas import AnalysisResult, Category, Severity

ANALYSIS = AnalysisResult.model_validate(
    {
        "summary": "found something",
        "findings": [
            {
                "severity": "critical",
                "category": "security",
                "title": "SQL injection in the query",
                "description": "The query interpolates user input.",
                "recommendation": "Use a parameterised query.",
                "location": {"line": 2},
            }
        ],
    }
)

CLEAN_CASE = DetectionCase(
    id="clean",
    language="python",
    source='def add(a, b):\n    """Add."""\n    return a + b\n',
    clean=True,
)


def _defect(
    *,
    id: str,
    line: int,
    categories: tuple[tuple[str, ...], ...],
    severity: Severity = Severity.CRITICAL,
    category: Category = Category.SECURITY,
) -> PlantedDefect:
    return PlantedDefect(
        id=id,
        summary="summary",
        category=category,
        severity=severity,
        line=line,
        keywords=categories,
        note="note",
    )


class TestReviewWithStatic:
    def test_produces_findings_without_any_model_call(self) -> None:
        case = DetectionCase(
            id="x",
            language="python",
            source="def f(bucket=[]):\n    return bucket\n",
            defects=(),
        )
        result = review_with_static(case)
        assert len(result.findings) == 1
        assert "utable" in result.findings[0].title

    def test_reports_a_bare_except(self) -> None:
        case = DetectionCase(
            id="x",
            language="python",
            source="def f():\n    try:\n        pass\n    except:\n        pass\n",
            defects=(),
        )
        titles = " ".join(f.title for f in review_with_static(case).findings)
        assert "except" in titles

    def test_finds_nothing_in_clean_code(self) -> None:
        assert review_with_static(CLEAN_CASE).findings == []

    def test_findings_carry_real_line_numbers(self) -> None:
        case = DetectionCase(
            id="x",
            language="python",
            source="def f(bucket=[]):\n    return bucket\n",
            defects=(),
        )
        finding = review_with_static(case).findings[0]
        assert finding.location is not None
        assert finding.location.line == 1

    def test_reports_a_syntax_error_as_no_findings(self) -> None:
        case = DetectionCase(id="x", language="python", source="def broken(:\n", defects=())
        assert review_with_static(case).findings == []

    def test_non_python_is_skipped(self) -> None:
        case = DetectionCase(id="x", language="javascript", source="function f(){}", defects=())
        assert review_with_static(case).findings == []


class TestRunBenchmark:
    def test_static_tier_needs_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        report = run_benchmark([Tier.STATIC], use_cache=False)
        assert report.scores
        assert report.total_calls == 0

    def test_reports_the_corpus_size(self) -> None:
        assert run_benchmark([Tier.STATIC], use_cache=False).corpus_size >= 5

    def test_scores_one_tier_per_request(self) -> None:
        report = run_benchmark([Tier.STATIC, Tier.STATIC], use_cache=False)
        assert len(report.scores) == 2

    def test_the_static_tier_scores_above_zero(self) -> None:
        """The corpus contains the patterns it detects.

        Without this the tier reads 0% for reasons about the corpus, not the
        tool, which is exactly the kind of misleading number to avoid.
        """
        score = run_benchmark([Tier.STATIC], use_cache=False).scores[0]
        assert score.recall is not None and score.recall > 0

    def test_the_static_tier_is_quiet_on_the_clean_case(self) -> None:
        score = run_benchmark([Tier.STATIC], use_cache=False).scores[0]
        assert score.clean_case_findings == 0

    def test_case_filter_restricts_the_run(self) -> None:
        report = run_benchmark([Tier.STATIC], case_filter=["insecure_sql"], use_cache=False)
        assert report.corpus_size == 1

    def test_an_unknown_case_filter_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no cases matched"):
            run_benchmark([Tier.STATIC], case_filter=["does-not-exist"], use_cache=False)

    def test_a_model_tier_without_a_key_reports_the_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One failed case must not end the run or crash the harness."""
        from ai_pair_engineer.services.llm import MissingAPIKeyError

        monkeypatch.setattr(
            "ai_pair_engineer.agents.base.ask_llm",
            lambda **kwargs: (_ for _ in ()).throw(MissingAPIKeyError("no key")),
        )
        report = run_benchmark([Tier.ANALYZER], use_cache=False)
        assert report.scores
        assert all(result.error for result in report.case_results)

    def test_a_broken_response_is_recorded_not_raised(self, mock_llm: Any) -> None:
        mock_llm.set("not json")
        report = run_benchmark([Tier.ANALYZER], use_cache=False)
        assert any(result.error for result in report.case_results)


class TestCache:
    def test_a_cached_result_is_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_llm: Any
    ) -> None:
        monkeypatch.setattr(runner_module, "CACHE_DIR", tmp_path)
        mock_llm.set(ANALYSIS.model_dump_json())

        run_benchmark([Tier.ANALYZER], model="m", use_cache=True)
        first = mock_llm.calls()
        run_benchmark([Tier.ANALYZER], model="m", use_cache=True)

        assert first == report_case_count()
        assert mock_llm.calls() == first

    def test_refresh_ignores_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_llm: Any
    ) -> None:
        monkeypatch.setattr(runner_module, "CACHE_DIR", tmp_path)
        mock_llm.set(ANALYSIS.model_dump_json())

        run_benchmark([Tier.ANALYZER], model="m", use_cache=True)
        run_benchmark([Tier.ANALYZER], model="m", use_cache=True, refresh=True)
        assert mock_llm.calls() == 2 * report_case_count()

    def test_editing_a_case_invalidates_its_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cached score against stale code is worse than no cache at all."""
        monkeypatch.setattr(runner_module, "CACHE_DIR", tmp_path)

        case = DetectionCase(id="c", language="python", source="x = 1\n", defects=())

        before = runner_module._fingerprint(case, Tier.ANALYZER, "m", 0.0)
        edited = DetectionCase(id="c", language="python", source="x = 2\n", defects=())
        after = runner_module._fingerprint(edited, Tier.ANALYZER, "m", 0.0)

        assert before != after

    def test_the_cache_key_covers_the_model_and_tier(self) -> None:
        case = DetectionCase(id="c", language="python", source="x = 1\n", defects=())
        keys = {
            runner_module._fingerprint(case, tier, model, temp)
            for tier in Tier
            for model in ("m1", "m2")
            for temp in (0.0, 0.7)
        }
        assert len(keys) == len(Tier) * 2 * 2

    def test_the_cache_key_covers_the_defect_annotations(self) -> None:
        """Re-annotating ground truth must invalidate a cached score."""
        one = DetectionCase(id="c", language="python", source="x = 1\n", defects=())
        two = DetectionCase(
            id="c",
            language="python",
            source="x = 1\n",
            defects=(_defect(id="d", line=1, categories=(("a",), ("b",))),),
        )
        assert runner_module._fingerprint(one, Tier.STATIC, "m", 0.0) != (
            runner_module._fingerprint(two, Tier.STATIC, "m", 0.0)
        )

    def test_no_cache_leaves_nothing_on_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_llm: Any
    ) -> None:
        monkeypatch.setattr(runner_module, "CACHE_DIR", tmp_path)
        run_benchmark([Tier.ANALYZER], model="m", use_cache=False)
        assert not list(tmp_path.glob("*.json"))

    def test_the_static_tier_never_writes_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deterministic and free, so caching it could only serve a stale result."""
        monkeypatch.setattr(runner_module, "CACHE_DIR", tmp_path)
        run_benchmark([Tier.STATIC], use_cache=True)
        assert not list(tmp_path.glob("*.json"))

    def test_a_corrupt_cache_entry_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_llm: Any
    ) -> None:
        monkeypatch.setattr(runner_module, "CACHE_DIR", tmp_path)
        run_benchmark([Tier.ANALYZER], model="m", use_cache=True)

        for path in tmp_path.glob("*.json"):
            path.write_text("{ not json", encoding="utf-8")

        mock_llm.set(ANALYSIS.model_dump_json())
        report = run_benchmark([Tier.ANALYZER], model="m", use_cache=True)
        assert not any(result.error for result in report.case_results)


def report_case_count() -> int:
    from benchmarks.corpus import load_cases

    return len(load_cases())


class TestTierOrder:
    def test_parses_all_into_benchmark_order(self) -> None:
        assert tier_order(["all"]) == [
            Tier.STATIC,
            Tier.NAIVE,
            Tier.ANALYZER,
            Tier.ANALYZER_STATIC,
        ]

    def test_returns_only_the_requested_tiers_in_canonical_order(self) -> None:
        assert tier_order(["analyzer+static", "static"]) == [
            Tier.STATIC,
            Tier.ANALYZER_STATIC,
        ]

    def test_unknown_names_are_dropped(self) -> None:
        assert tier_order(["nonsense"]) == []

    def test_uses_model_flag_is_correct(self) -> None:
        assert not Tier.STATIC.uses_model
        assert Tier.NAIVE.uses_model
        assert Tier.ANALYZER.uses_model
        assert Tier.ANALYZER_STATIC.uses_model


def _report(
    *,
    scores: list[TierScore] | None = None,
    regressions: list[RegressionResult] | None = None,
) -> RunReport:
    return RunReport(
        model="vendor/model",
        temperature=0.0,
        corpus_size=7,
        scores=scores if scores is not None else [],
        case_results=[],
        regression_results=regressions or [],
        total_tokens=1234,
        total_cost_usd=0.0123,
        total_calls=7,
    )


class TestReportRendering:
    def test_renders_a_headline_table(self) -> None:
        report = _report(
            scores=[
                TierScore(
                    tier=Tier.STATIC,
                    cases=7,
                    planted=22,
                    detected=2,
                    findings=2,
                    matched_findings=2,
                )
            ]
        )
        assert "| Tier | Recall |" in render_markdown(report)

    def test_shows_measured_cost_and_tokens(self) -> None:
        text = render_markdown(_report())
        assert "1234" in text
        assert "vendor/model" in text

    def test_unpriced_cost_says_so_rather_than_showing_zero(self) -> None:
        unpriced = RunReport(model="local", temperature=0.0, corpus_size=1)
        assert "not priced" in render_markdown(unpriced)

    def test_a_priced_run_shows_dollars(self) -> None:
        assert "$0.01" in render_markdown(_report())

    def test_absent_metrics_render_as_n_a(self) -> None:
        report = _report(scores=[TierScore(tier=Tier.STATIC, cases=1)])
        text = render_markdown(report)
        assert "n/a" in text

    def test_states_that_groundedness_is_not_precision(self) -> None:
        """The caveat has to travel with the number."""
        text = render_markdown(_report(scores=[TierScore(tier=Tier.STATIC)]))
        assert "not precision" in text

    def test_warns_about_the_corpus_conflict_of_interest(self) -> None:
        assert "conflict of interest" in render_markdown(_report())

    def test_warns_about_confidence_intervals(self) -> None:
        assert "confidence interval" in render_markdown(_report())

    def test_regression_section_requires_both_conditions(self) -> None:
        text = render_markdown(
            _report(
                regressions=[
                    RegressionResult(
                        case_id="r",
                        tier=Tier.ANALYZER_STATIC,
                        approved=True,
                        mentioned=False,
                        verdict="approve",
                        score=90,
                        reasoning="looks fine",
                    )
                ]
            )
        )
        assert "Caught **0/1**" in text
        assert "Approved anyway" in text

    def test_regression_section_counts_a_full_catch(self) -> None:
        text = render_markdown(
            _report(
                regressions=[
                    RegressionResult(
                        case_id="r",
                        tier=Tier.ANALYZER_STATIC,
                        approved=False,
                        mentioned=True,
                        verdict="reject",
                        score=20,
                        reasoning="the boundary value is now excluded",
                    )
                ]
            )
        )
        assert "Caught **1/1**" in text

    def test_json_output_round_trips(self) -> None:
        report = _report(scores=[TierScore(tier=Tier.STATIC, planted=3, detected=1)])
        payload = json.loads(render_json(report))
        assert payload["model"] == "vendor/model"
        assert payload["scores"][0]["detected"] == 1

    def test_json_omits_per_case_detail_unless_verbose(self) -> None:
        report = _report()
        assert "case_results" not in json.loads(render_json(report))
        assert "case_results" in json.loads(render_json(report, verbose=True))

    def test_json_includes_totals(self) -> None:
        payload = json.loads(render_json(_report()))
        assert payload["totals"]["tokens"] == 1234

    def test_output_is_ascii_safe(self) -> None:
        """Windows consoles mojibake non-ASCII placeholders."""
        render_markdown(_report(scores=[TierScore(tier=Tier.STATIC, cases=1)]))


class TestRegressionResult:
    def test_caught_requires_rejection_and_a_named_reason(self) -> None:
        both = RegressionResult(
            case_id="r",
            tier=Tier.ANALYZER_STATIC,
            approved=False,
            mentioned=True,
            verdict="reject",
            score=10,
            reasoning="boundary changed",
        )
        assert both.caught

    def test_approving_is_never_a_catch(self) -> None:
        result = RegressionResult(
            case_id="r",
            tier=Tier.ANALYZER_STATIC,
            approved=True,
            mentioned=True,
            verdict="approve",
            score=90,
            reasoning="hmm",
        )
        assert not result.caught

    def test_noticing_without_naming_is_not_a_catch(self) -> None:
        result = RegressionResult(
            case_id="r",
            tier=Tier.ANALYZER_STATIC,
            approved=False,
            mentioned=False,
            verdict="needs_review",
            score=50,
            reasoning="something",
        )
        assert not result.caught

    def test_serialises_the_caught_flag(self) -> None:
        result = RegressionResult(
            case_id="r",
            tier=Tier.STATIC,
            approved=False,
            mentioned=True,
            verdict="reject",
            score=10,
            reasoning="x",
        )
        assert result.to_dict()["caught"] is True


class TestCaseResultSerialisation:
    def test_detected_is_deduplicated(self) -> None:
        outcome = CaseResult(
            case_id="c",
            tier=Tier.STATIC,
            matches=[Match("a", "t1", MatchRule.LOCATED), Match("a", "t2", MatchRule.SEMANTIC)],
        )
        assert outcome.detected == {"a"}

    def test_serialises_finding_line_numbers(self) -> None:
        outcome = CaseResult(
            case_id="c",
            tier=Tier.STATIC,
            findings=ANALYSIS.findings,
            matches=[],
        )
        assert outcome.to_dict()["findings"][0]["line"] == 2

    def test_serialises_an_error(self) -> None:
        outcome = CaseResult(case_id="c", tier=Tier.STATIC, error="boom")
        assert outcome.to_dict()["error"] == "boom"
