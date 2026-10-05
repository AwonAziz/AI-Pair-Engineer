"""Tests for end-to-end pipeline wiring.

The model boundary is mocked once per test and the assertions are about
orchestration: that the right context reaches each stage, that failures surface
with useful messages, and that no real network call happens.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from ai_pair_engineer.agents import StageError
from ai_pair_engineer.models.schemas import Severity
from ai_pair_engineer.pipeline import PipelineResult, UnsupportedLanguageError, run_pipeline
from ai_pair_engineer.services.llm import LLMError, MissingAPIKeyError

LANGUAGES = ("python", "javascript", "typescript", "java")

# Every stage calls `ask_llm` through the shared base class, so this is the one
# place a test needs to patch to intercept all four.
PATCH_TARGET = "ai_pair_engineer.agents.base.ask_llm"


@pytest.fixture
def llm(canned: dict[str, str]) -> Iterator[Any]:
    """Patch the LLM boundary, routing each stage to its canned response.

    Stage detection keys off the system prompt, so a change to a prompt also
    changes the routing. That is deliberate: it fails loudly rather than
    silently feeding the analyzer a reviewer's response.
    """

    def dispatch(**call: Any) -> str:
        prompt = call["system_prompt"].lower()
        if "test engineer" in prompt:
            return canned["tests"]
        if "senior engineer improving" in prompt:
            return canned["refactor"]
        if "adversarial reviewer" in prompt:
            return canned["review"]
        return canned["analysis"]

    with patch(PATCH_TARGET, side_effect=dispatch) as mock:
        yield mock


class TestHappyPath:
    def test_returns_every_stage_result(self, sample_source: str, llm: Any) -> None:
        result = run_pipeline("python", sample_source)

        assert isinstance(result, PipelineResult)
        assert len(result.analysis.findings) == 3
        assert len(result.tests.tests) == 2
        assert result.refactor.refactored_code
        assert result.review.verdict == "needs_review"

    def test_runs_four_stages(self, sample_source: str, llm: Any) -> None:
        run_pipeline("python", sample_source)
        assert llm.call_count == 4

    def test_records_each_stage_in_order(self, sample_source: str, llm: Any) -> None:
        result = run_pipeline("python", sample_source)
        assert [stage.stage for stage in result.trace.stages] == [
            "Code Analyzer",
            "Test Engineer",
            "Refactoring Engineer",
            "Final Reviewer",
        ]

    def test_attaches_the_static_report(self, sample_source: str, llm: Any) -> None:
        result = run_pipeline("python", sample_source)
        assert result.static_report.function_count == 1
        assert result.static_report.supported

    def test_computes_the_quality_score(self, sample_source: str, llm: Any) -> None:
        result = run_pipeline("python", sample_source)
        assert result.quality_score == result.analysis.quality_score()

    def test_reports_the_highest_severity(self, sample_source: str, llm: Any) -> None:
        assert run_pipeline("python", sample_source).highest_severity is Severity.HIGH


class TestLanguageHandling:
    @pytest.mark.parametrize("language", LANGUAGES)
    def test_accepts_every_supported_language(
        self, language: str, sample_source: str, llm: Any
    ) -> None:
        assert run_pipeline(language, sample_source).language == language

    def test_normalises_case(self, sample_source: str, llm: Any) -> None:
        assert run_pipeline("Python", sample_source).language == "python"

    def test_rejects_an_unknown_language(self, sample_source: str) -> None:
        with pytest.raises(UnsupportedLanguageError, match="unsupported language"):
            run_pipeline("cobol", sample_source)

    def test_rejects_an_empty_language(self, sample_source: str) -> None:
        with pytest.raises(UnsupportedLanguageError):
            run_pipeline("", sample_source)

    def test_non_python_runs_without_static_evidence(self, llm: Any) -> None:
        result = run_pipeline("javascript", "function f() { return 1; }")
        assert result.static_report.supported is False


class TestInputValidation:
    def test_rejects_empty_source(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            run_pipeline("python", "")

    def test_rejects_whitespace_source(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            run_pipeline("python", "\n\n\t")

    def test_language_is_validated_before_any_model_call(self, sample_source: str) -> None:
        with (
            patch(PATCH_TARGET) as mock,
            pytest.raises(UnsupportedLanguageError),
        ):
            run_pipeline("fortran", sample_source)
        mock.assert_not_called()

    def test_empty_source_is_rejected_before_any_model_call(self) -> None:
        with (
            patch(PATCH_TARGET) as mock,
            pytest.raises(ValueError),
        ):
            run_pipeline("python", "")
        mock.assert_not_called()

    def test_unparseable_source_still_produces_a_result(self, llm: Any) -> None:
        """A syntax error is something to report, not a reason to crash."""
        result = run_pipeline("python", "def broken(:\n")
        assert result.static_report.syntax_error is not None


class TestFailurePropagation:
    def test_a_failing_stage_names_itself(self, sample_source: str) -> None:
        with (
            patch(PATCH_TARGET, return_value="garbage"),
            pytest.raises(StageError, match="Code Analyzer"),
        ):
            run_pipeline("python", sample_source)

    def test_a_missing_key_surfaces_as_a_config_error(self, sample_source: str) -> None:
        with (
            patch(
                "ai_pair_engineer.services.llm.get_client",
                side_effect=MissingAPIKeyError("no key"),
            ),
            pytest.raises(LLMError, match="no key"),
        ):
            run_pipeline("python", sample_source)

    def test_a_transport_failure_propagates(self, sample_source: str) -> None:
        with (
            patch(
                PATCH_TARGET,
                side_effect=LLMError("provider exploded"),
            ),
            pytest.raises(LLMError, match="provider exploded"),
        ):
            run_pipeline("python", sample_source)

    def test_progress_reports_the_failing_stage(self, sample_source: str) -> None:
        events: list[tuple[str, str]] = []
        with (
            patch(PATCH_TARGET, return_value="garbage"),
            pytest.raises(StageError),
        ):
            run_pipeline("python", sample_source, on_progress=lambda s, e: events.append((s, e)))
        assert [stage for stage, _ in events] == ["Code Analyzer", "Code Analyzer"]

    def test_progress_reports_start_and_finish(self, sample_source: str, llm: Any) -> None:
        events: list[tuple[str, str]] = []
        run_pipeline("python", sample_source, on_progress=lambda s, e: events.append((s, e)))
        statuses = [status for _, status in events]
        assert statuses.count("started") == 4
        assert all(status.startswith("finished") for status in statuses if status != "started")


class TestContextScoping:
    def test_the_tester_never_sees_the_refactored_code(self, sample_source: str, llm: Any) -> None:
        run_pipeline("python", sample_source)
        tester = _call_for(llm, "test engineer")
        assert "results.append(user)" not in tester.kwargs["user_prompt"]

    def test_the_refactorer_never_sees_the_generated_tests(
        self, sample_source: str, llm: Any
    ) -> None:
        run_pipeline("python", sample_source)
        refactor = _call_for(llm, "senior engineer improving")
        assert "def test_missing_age" not in refactor.kwargs["user_prompt"]

    def test_the_analyzer_sees_static_evidence(self, sample_source: str, llm: Any) -> None:
        run_pipeline("python", sample_source)
        assert "Deterministic static analysis" in llm.call_args_list[0].kwargs["user_prompt"]

    def test_the_reviewer_sees_both_versions(self, sample_source: str, llm: Any) -> None:
        run_pipeline("python", sample_source)
        reviewer = _call_for(llm, "adversarial reviewer")
        assert "ORIGINAL CODE" in reviewer.kwargs["user_prompt"]
        assert "REFACTORED CODE" in reviewer.kwargs["user_prompt"]

    def test_a_model_override_reaches_every_stage(self, sample_source: str, llm: Any) -> None:
        run_pipeline("python", sample_source, model="vendor/model-x")
        assert {call.kwargs["model"] for call in llm.call_args_list} == {"vendor/model-x"}

    def test_a_precomputed_static_report_is_reused(self, sample_source: str, llm: Any) -> None:
        from ai_pair_engineer.static import analyze_source

        report = analyze_source(sample_source, "python")
        result = run_pipeline("python", sample_source, static_report=report)
        assert result.static_report is report


class TestSerialisation:
    def test_produces_json_serialisable_output(self, sample_source: str, llm: Any) -> None:
        payload = run_pipeline("python", sample_source).to_dict()
        assert json.loads(json.dumps(payload))["quality_score"] >= 0

    def test_omits_code_unless_requested(self, sample_source: str, llm: Any) -> None:
        result = run_pipeline("python", sample_source)
        assert "refactored_code" not in result.to_dict()
        assert "refactored_code" in result.to_dict(include_code=True)

    def test_includes_finding_line_numbers(self, sample_source: str, llm: Any) -> None:
        payload = run_pipeline("python", sample_source).to_dict()
        assert payload["findings"][0]["line"] == 3

    def test_includes_usage_accounting(self, sample_source: str, llm: Any) -> None:
        payload = run_pipeline("python", sample_source).to_dict()
        assert len(payload["usage"]["stages"]) == 4


def _call_for(mock: Any, prompt_fragment: str) -> Any:
    """Find the recorded call whose system prompt contains a fragment."""
    for call in mock.call_args_list:
        if prompt_fragment in call.kwargs["system_prompt"].lower():
            return call
    raise AssertionError(f"no stage prompt contained {prompt_fragment!r}")
