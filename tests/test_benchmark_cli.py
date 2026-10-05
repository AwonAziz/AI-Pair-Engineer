"""Tests for the benchmark command line.

This is the surface a user touches first, and the parts most likely to break
silently are the exit codes and the ordering of validation against spending
money. A harness that runs 300 model calls and only then reports a broken
corpus is worse than one that refuses to start.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from benchmarks.__main__ import main
from benchmarks.models import Tier


class TestValidateCommand:
    def test_the_shipped_corpus_validates(self, capsys: Any) -> None:
        assert main(["validate"]) == 0
        assert "Corpus is consistent" in capsys.readouterr().out

    def test_reports_the_case_and_defect_counts(self, capsys: Any) -> None:
        main(["validate"])
        out = capsys.readouterr().out
        assert "planted defect" in out
        assert "regression case" in out

    def test_needs_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert main(["validate"]) == 0

    def test_a_subcommand_is_required(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_an_unknown_subcommand_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            main(["nonsense"])


class TestRunArgumentValidation:
    def test_rejects_an_unknown_tier(self, capsys: Any) -> None:
        assert main(["run", "--tiers", "nonsense"]) == 2
        assert "unknown tier" in capsys.readouterr().err

    def test_names_the_valid_tiers_when_rejecting(self, capsys: Any) -> None:
        main(["run", "--tiers", "nonsense"])
        assert "analyzer+static" in capsys.readouterr().err

    def test_rejects_an_empty_tier_list(self, capsys: Any) -> None:
        assert main(["run", "--tiers", " , "]) == 2

    def test_rejects_an_unknown_format(self) -> None:
        with pytest.raises(SystemExit):
            main(["run", "--tiers", "static", "--format", "xml"])

    def test_the_static_tier_runs_without_a_key(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert main(["run", "--tiers", "static"]) == 0

    def test_a_model_tier_without_a_key_exits_with_the_config_code(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert main(["run", "--tiers", "analyzer"]) == 3
        assert "OPENROUTER_API_KEY" in capsys.readouterr().err

    def test_the_config_error_names_the_free_tier(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        main(["run", "--tiers", "analyzer"])
        assert "static" in capsys.readouterr().err

    def test_an_unknown_case_filter_exits_with_usage(self, capsys: Any) -> None:
        assert main(["run", "--tiers", "static", "--cases", "nope"]) == 2
        assert "no cases matched" in capsys.readouterr().err


class TestCorpusValidationPrecedesSpending:
    def test_a_broken_corpus_stops_the_run_before_any_call(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """Money must not be spent on a corpus whose ground truth is broken."""
        import benchmarks.__main__ as entry

        monkeypatch.setattr(entry, "validate_corpus", lambda: (["case 'x' is broken"], 1, 1))

        called = {"n": 0}

        def _never(*args: Any, **kwargs: Any) -> None:
            called["n"] += 1
            raise AssertionError("must not reach the model")

        monkeypatch.setattr("ai_pair_engineer.agents.base.ask_llm", _never)
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

        assert main(["run", "--tiers", "analyzer"]) == 2
        assert called["n"] == 0
        assert "broken" in capsys.readouterr().err

    def test_a_corpus_failure_is_reported_as_usage(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """A malformed corpus on disk is the caller's problem, not a crash."""
        import benchmarks.__main__ as entry
        from benchmarks.corpus import CorpusError

        def _explode() -> Any:
            raise CorpusError("bad toml")

        monkeypatch.setattr(entry, "validate_corpus", _explode)
        assert main(["run", "--tiers", "static"]) == 2
        assert "bad toml" in capsys.readouterr().err


class TestOutput:
    def test_writes_markdown_to_stdout_by_default(self, capsys: Any) -> None:
        main(["run", "--tiers", "static"])
        assert "# AI Pair Engineer benchmark" in capsys.readouterr().out

    def test_writes_json_when_asked(self, capsys: Any) -> None:
        main(["run", "--tiers", "static", "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["scores"][0]["tier"] == "static"

    def test_writes_to_a_file_when_asked(self, tmp_path: Path, capsys: Any) -> None:
        target = tmp_path / "report.md"
        assert main(["run", "--tiers", "static", "--out", str(target)]) == 0
        assert "Report written" in capsys.readouterr().out
        assert target.read_text(encoding="utf-8").startswith("# AI Pair Engineer")

    def test_the_breakdown_renders_for_a_single_tier(self, capsys: Any) -> None:
        """A one-column breakdown is how you spot a category blind spot."""
        main(["run", "--tiers", "static", "--breakdown"])
        out = capsys.readouterr().out
        assert "## Breakdown" in out
        assert "| Category |" in out
        assert "| Severity |" in out

    def test_verbose_adds_per_case_detail(self, capsys: Any) -> None:
        main(["run", "--tiers", "static", "--verbose"])
        assert "Per-case detail" in capsys.readouterr().out

    def test_verbose_progress_is_printed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.setattr(
            "benchmarks.runner.run_benchmark",
            lambda *a, **k: _empty_report(),
        )
        main(["run", "--tiers", "static", "--verbose"])
        assert "[static only]" in capsys.readouterr().out


def _empty_report() -> Any:
    from benchmarks.models import RunReport

    return RunReport(model="m", temperature=0.0, corpus_size=0)


class TestRegressionInvocation:
    """The regression path is what justifies the whole four-stage design."""

    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Intercept the runner where the CLI imported it.

        The CLI does ``from benchmarks.runner import run_benchmark``, so
        patching the runner module would not be seen here.
        """
        import benchmarks.__main__ as entry

        captured: dict[str, Any] = {}

        def _fake_run(tiers: list[Any], **kwargs: Any) -> Any:
            captured.update(kwargs)
            captured["tiers"] = tiers
            return _empty_report()

        monkeypatch.setattr(entry, "run_benchmark", _fake_run)
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
        return captured

    def test_the_baseline_flag_uses_the_plain_reviewer(self, _capture: dict[str, Any]) -> None:
        main(["run", "--tiers", "analyzer+static", "--regressions-baseline"])
        assert _capture["naive_reviewer"] is True
        assert _capture["run_regressions"] is True

    def test_the_adversarial_flag_is_the_default(self, _capture: dict[str, Any]) -> None:
        main(["run", "--tiers", "analyzer+static", "--regressions"])
        assert _capture["naive_reviewer"] is False

    def test_the_model_override_is_forwarded(self, _capture: dict[str, Any]) -> None:
        main(["run", "--tiers", "static", "--model", "vendor/x"])
        assert _capture["model"] == "vendor/x"

    def test_the_temperature_override_is_forwarded(self, _capture: dict[str, Any]) -> None:
        main(["run", "--tiers", "static", "--temperature", "0.7"])
        assert _capture["temperature"] == pytest.approx(0.7)

    def test_no_cache_is_forwarded(self, _capture: dict[str, Any]) -> None:
        main(["run", "--tiers", "static", "--no-cache"])
        assert _capture["use_cache"] is False

    def test_regressions_need_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert main(["run", "--tiers", "static", "--regressions"]) == 3


class TestRunRegressionCase:
    """The regression path is what justifies the whole four-stage design."""

    def _case(self) -> Any:
        from benchmarks.models import RegressionCase

        return RegressionCase(
            id="r",
            original="def f(x):\n    return x\n",
            refactored="def f(x):\n    return x + 0\n",
            change="adds zero",
            expected=("boundary",),
        )

    def test_counts_a_rejection_that_names_the_change(self, mock_llm: Any) -> None:
        from benchmarks.runner import run_regression_case

        mock_llm.set(_review_json(approved=False, summary="The boundary case changed."))
        result = run_regression_case(
            self._case(),
            tier=Tier.ANALYZER_STATIC,
            model="m",
            temperature=0.0,
            naive_reviewer=False,
        )
        assert result.caught

    def test_an_approval_is_never_a_catch(self, mock_llm: Any) -> None:
        from benchmarks.runner import run_regression_case

        mock_llm.set(_review_json(approved=True, summary="Looks equivalent to me."))
        result = run_regression_case(
            self._case(),
            tier=Tier.ANALYZER_STATIC,
            model="m",
            temperature=0.0,
            naive_reviewer=False,
        )
        assert not result.caught
        assert result.approved

    def test_rejecting_without_naming_the_change_is_not_a_catch(self, mock_llm: Any) -> None:
        from benchmarks.runner import run_regression_case

        mock_llm.set(_review_json(approved=False, summary="Some style concerns."))
        result = run_regression_case(
            self._case(),
            tier=Tier.ANALYZER_STATIC,
            model="m",
            temperature=0.0,
            naive_reviewer=False,
        )
        assert not result.approved
        assert not result.mentioned

    def test_matches_a_phrase_in_remaining_issues(self, mock_llm: Any) -> None:
        from benchmarks.runner import run_regression_case

        mock_llm.set(
            _review_json(
                approved=False,
                summary="Something differs.",
                remaining_issues=["The boundary value is now excluded."],
            )
        )
        result = run_regression_case(
            self._case(),
            tier=Tier.ANALYZER_STATIC,
            model="m",
            temperature=0.0,
            naive_reviewer=False,
        )
        assert result.mentioned

    def test_matches_a_phrase_in_the_recommendation(self, mock_llm: Any) -> None:
        from benchmarks.runner import run_regression_case

        mock_llm.set(
            _review_json(approved=False, summary="Differs.", recommendation="Fix the boundary.")
        )
        result = run_regression_case(
            self._case(),
            tier=Tier.ANALYZER_STATIC,
            model="m",
            temperature=0.0,
            naive_reviewer=False,
        )
        assert result.mentioned

    def test_a_provider_failure_is_recorded_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchmarks.runner import run_regression_case

        from ai_pair_engineer.services.llm import LLMError

        monkeypatch.setattr(
            "ai_pair_engineer.agents.base.ask_llm",
            lambda **kwargs: (_ for _ in ()).throw(LLMError("upstream 503")),
        )
        result = run_regression_case(
            self._case(),
            tier=Tier.ANALYZER_STATIC,
            model="m",
            temperature=0.0,
            naive_reviewer=False,
        )
        assert result.error
        assert not result.caught

    def test_the_baseline_prompt_is_used_when_asked(self, mock_llm: Any) -> None:
        from benchmarks.runner import run_regression_case

        mock_llm.set(_review_json(approved=False, summary="The boundary case changed."))
        result = run_regression_case(
            self._case(), tier=Tier.ANALYZER_STATIC, model="m", temperature=0.0, naive_reviewer=True
        )
        assert result.caught

    def test_records_token_usage(self, mock_llm: Any) -> None:
        from benchmarks.runner import run_regression_case

        mock_llm.set(_review_json(approved=False, summary="The boundary case changed."))
        result = run_regression_case(
            self._case(),
            tier=Tier.ANALYZER_STATIC,
            model="m",
            temperature=0.0,
            naive_reviewer=False,
        )
        assert result.tokens >= 0
        assert result.cost_usd >= 0


def _review_json(
    *,
    approved: bool,
    summary: str,
    remaining_issues: list[str] | None = None,
    recommendation: str = "Review the diff.",
) -> str:
    return json.dumps(
        {
            "approved": approved,
            "verdict": "approve" if approved else "reject",
            "score": 90 if approved else 20,
            "summary": summary,
            "strengths": [],
            "remaining_issues": remaining_issues or [],
            "regression_risk": "low" if approved else "high",
            "recommendation": recommendation,
        }
    )


class TestCaseFilterNarrowsTheCorpus:
    def test_restricts_to_one_case(self, capsys: Any) -> None:
        main(["run", "--tiers", "static", "--cases", "insecure_sql", "--format", "json"])
        assert json.loads(capsys.readouterr().out)["corpus_size"] == 1

    def test_reports_the_full_corpus_size(self, capsys: Any) -> None:
        main(["run", "--tiers", "static", "--format", "json"])
        assert json.loads(capsys.readouterr().out)["corpus_size"] >= 5


class TestTierParsing:
    @pytest.fixture(autouse=True)
    def _key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    def test_all_expands_to_every_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import benchmarks.__main__ as entry

        captured: dict[str, Any] = {}

        def _fake_run(tiers: list[Any], **kwargs: Any) -> Any:
            captured["tiers"] = tiers
            return _empty_report()

        monkeypatch.setattr(entry, "run_benchmark", _fake_run)
        assert main(["run", "--tiers", "all"]) == 0
        assert captured["tiers"] == list(Tier)

    def test_all_refuses_without_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert main(["run", "--tiers", "all"]) == 3

    def test_a_mixed_list_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import benchmarks.__main__ as entry

        captured: dict[str, Any] = {}

        def _fake_run(tiers: list[Any], **kwargs: Any) -> Any:
            captured["tiers"] = tiers
            return _empty_report()

        monkeypatch.setattr(entry, "run_benchmark", _fake_run)
        main(["run", "--tiers", "static,analyzer+static"])
        assert captured["tiers"] == [Tier.STATIC, Tier.ANALYZER_STATIC]

    def test_requested_order_is_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reported in benchmark order so two runs are comparable."""
        import benchmarks.__main__ as entry

        captured: dict[str, Any] = {}

        def _fake_run(tiers: list[Any], **kwargs: Any) -> Any:
            captured["tiers"] = tiers
            return _empty_report()

        monkeypatch.setattr(entry, "run_benchmark", _fake_run)
        main(["run", "--tiers", "analyzer+static,static"])
        assert captured["tiers"] == [Tier.STATIC, Tier.ANALYZER_STATIC]
