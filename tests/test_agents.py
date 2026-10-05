"""Tests for the four pipeline stages.

Model responses are patched at the module boundary each stage imports from.
Nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ai_pair_engineer.agents import (
    AnalyzerAgent,
    RefactorAgent,
    ReviewerAgent,
    StageError,
    TesterAgent,
    analyze_code,
    generate_tests,
    refactor_code,
    review_refactor,
)
from ai_pair_engineer.models.schemas import (
    AnalysisResult,
    Category,
    FinalReview,
    RefactorResult,
    Severity,
    TestCase,
    TestResult,
)

# Stand-in for the refactorer's output, used wherever the reviewer needs a
# second version of the code to compare against.
REFACTORED = "def process_users(users):\n    return []\n"

# Every stage calls `ask_llm` through the shared base class, so this single
# patch target covers all four without reaching into each module.
_PATCH_TARGET = "ai_pair_engineer.agents.base.ask_llm"


def _patch(canned: dict[str, str], key: str):
    return patch(_PATCH_TARGET, return_value=canned[key])


class TestAnalyzer:
    def test_returns_a_validated_result(self, sample_source: str, canned: dict[str, str]) -> None:
        with _patch(canned, "analysis"):
            result = analyze_code("python", sample_source)
        assert isinstance(result, AnalysisResult)
        assert len(result.findings) == 3

    def test_preserves_locations(self, sample_source: str, canned: dict[str, str]) -> None:
        with _patch(canned, "analysis"):
            result = analyze_code("python", sample_source)
        assert result.findings[0].location is not None
        assert result.findings[0].location.line == 3

    def test_rejects_empty_source(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            analyze_code("python", "")

    def test_rejects_whitespace_only_source(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            analyze_code("python", "   \n\t  ")

    def test_passes_static_evidence_to_the_model(
        self, sample_source: str, canned: dict[str, str]
    ) -> None:
        with patch("ai_pair_engineer.agents.base.ask_llm", return_value=canned["analysis"]) as mock:
            analyze_code("python", sample_source)
        prompt = mock.call_args.kwargs["user_prompt"]
        assert "Deterministic static analysis" in prompt
        assert "functions: 1" in prompt

    def test_asks_for_a_second_attempt_after_a_bad_response(
        self, sample_source: str, canned: dict[str, str]
    ) -> None:
        with patch(
            "ai_pair_engineer.agents.base.ask_llm",
            side_effect=["not json at all", canned["analysis"]],
        ) as mock:
            assert analyze_code("python", sample_source).summary
        assert mock.call_count == 2

    def test_the_correction_prompt_mentions_the_schema(
        self, sample_source: str, canned: dict[str, str]
    ) -> None:
        with patch(
            "ai_pair_engineer.agents.base.ask_llm",
            side_effect=["garbage", canned["analysis"]],
        ) as mock:
            analyze_code("python", sample_source)
        assert "did not match the required schema" in mock.call_args.kwargs["system_prompt"]

    def test_raises_after_two_bad_responses(self, sample_source: str) -> None:
        with (
            patch("ai_pair_engineer.agents.base.ask_llm", return_value="nonsense"),
            pytest.raises(StageError, match="Code Analyzer"),
        ):
            analyze_code("python", sample_source)

    def test_gives_up_after_exactly_two_attempts(self, sample_source: str) -> None:
        with (
            patch("ai_pair_engineer.agents.base.ask_llm", return_value="nonsense") as mock,
            pytest.raises(StageError),
        ):
            analyze_code("python", sample_source)
        assert mock.call_count == 2

    def test_forwards_a_model_override(self, sample_source: str, canned: dict[str, str]) -> None:
        with patch("ai_pair_engineer.agents.base.ask_llm", return_value=canned["analysis"]) as mock:
            AnalyzerAgent("python", sample_source, model="some/model").run()
        assert mock.call_args.kwargs["model"] == "some/model"


class TestTester:
    def test_returns_a_validated_result(
        self, sample_source: str, analysis_result: AnalysisResult, canned: dict[str, str]
    ) -> None:
        with _patch(canned, "tests"):
            result = generate_tests("python", sample_source, analysis_result.findings)
        assert isinstance(result, TestResult)
        assert len(result.tests) == 2

    def test_rejects_empty_source(self, analysis_result: AnalysisResult) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_tests("python", "", analysis_result.findings)

    def test_does_not_forward_style_findings(self, sample_source: str) -> None:
        """A readability finding must not reach the test author.

        Style has no failure path, so forwarding it wastes context and invites
        tests that assert on formatting.
        """
        prompt = TesterAgent("python", sample_source, _findings()).build_user_prompt()
        assert "Inconsistent key naming" not in prompt

    def test_forwards_failure_findings(self, sample_source: str) -> None:
        prompt = TesterAgent("python", sample_source, _findings()).build_user_prompt()
        assert "Unvalidated dict access" in prompt
        assert "Shell injection" in prompt

    def test_prompt_includes_the_error_handling_finding(
        self, sample_source: str, canned: dict[str, str]
    ) -> None:
        agent = TesterAgent("python", sample_source, _findings())
        prompt = agent.build_user_prompt()
        assert "Unvalidated dict access" in prompt

    def test_prompt_lists_functions_needing_coverage(
        self, sample_source: str, canned: dict[str, str]
    ) -> None:
        prompt = TesterAgent("python", sample_source).build_user_prompt()
        assert "process_users" in prompt

    def test_works_with_no_findings(self, sample_source: str, canned: dict[str, str]) -> None:
        with patch("ai_pair_engineer.agents.base.ask_llm", return_value=canned["tests"]):
            assert isinstance(generate_tests("python", sample_source, []), TestResult)


class TestRefactor:
    def test_returns_the_new_source(self, sample_source: str, canned: dict[str, str]) -> None:
        with _patch(canned, "refactor"):
            result = refactor_code("python", sample_source, [])
        assert isinstance(result, RefactorResult)
        assert "def process_users" in result.refactored_code

    def test_rejects_empty_source(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            refactor_code("python", "", [])

    def test_receives_structure_findings(self, sample_source: str) -> None:
        prompt = RefactorAgent("python", sample_source, _findings()).build_user_prompt()
        assert "Deeply nested conditionals" in prompt

    def test_does_not_receive_failure_findings(self, sample_source: str) -> None:
        """Passing a security finding here invites a behaviour change mid-refactor.

        A correctness fix bundled into a refactor cannot be reviewed, which is
        exactly the change a reviewer least wants to verify.
        """
        prompt = RefactorAgent("python", sample_source, _findings()).build_user_prompt()
        assert "Shell injection" not in prompt
        assert "Unvalidated dict access" not in prompt

    def test_enforces_behaviour_preservation_in_the_prompt(self, sample_source: str) -> None:
        prompt = RefactorAgent("python", sample_source).build_user_prompt()
        assert "Preserve behaviour exactly" in prompt


class TestReviewer:
    def test_returns_a_verdict(self, sample_source: str, canned: dict[str, str]) -> None:
        with _patch(canned, "review"):
            result = review_refactor(sample_source, "def f():\n    pass", [], [])
        assert isinstance(result, FinalReview)
        assert result.approved is False
        assert result.verdict == "needs_review"

    def test_rejects_empty_original(self) -> None:
        with pytest.raises(ValueError, match="Original code cannot be empty"):
            review_refactor("", "def f(): pass", [], [])

    def test_rejects_empty_refactor(self) -> None:
        with pytest.raises(ValueError, match="Refactored code cannot be empty"):
            review_refactor("def f(): pass", "", [], [])

    def test_runs_at_zero_temperature(self, sample_source: str, canned: dict[str, str]) -> None:
        with patch("ai_pair_engineer.agents.base.ask_llm", return_value=canned["review"]) as mock:
            ReviewerAgent(sample_source, "def f():\n    pass").run()
        assert mock.call_args.kwargs["temperature"] == 0.0

    def test_surfaces_high_severity_findings_whatever_their_category(
        self, sample_source: str
    ) -> None:
        """The old filter compared category names against severity names.

        That silently dropped every high finding outside the security category,
        so the reviewer never saw the very defects it needed to check.
        """
        findings = [_finding(category=Category.COMPLEXITY, severity=Severity.CRITICAL)]
        agent = ReviewerAgent(sample_source, REFACTORED, [], findings)
        assert "Deeply nested conditionals" in agent.build_user_prompt()

    def test_omits_low_severity_findings(self, sample_source: str) -> None:
        findings = [_finding(severity=Severity.LOW)]
        agent = ReviewerAgent(sample_source, REFACTORED, [], findings)
        assert "Deeply nested conditionals" not in agent.build_user_prompt()

    def test_includes_test_names_but_not_their_code(self, sample_source: str) -> None:
        """Test bodies would dominate the prompt without helping the comparison."""
        tests = [TestCase(name="test_blank", purpose="blank input", test_code="SECRET_BODY")]
        agent = ReviewerAgent(sample_source, REFACTORED, generated_tests=tests)
        prompt = agent.build_user_prompt()
        assert "test_blank" in prompt
        assert "SECRET_BODY" not in prompt

    def test_is_told_not_to_approve_on_style_alone(self, sample_source: str) -> None:
        agent = ReviewerAgent(sample_source, REFACTORED)
        assert "stylistic difference is not a reason to approve" in agent.build_user_prompt()


class TestPromptLoading:
    @pytest.mark.parametrize("name", ["analyzer", "tester", "refactor", "reviewer"])
    def test_each_prompt_loads(self, name: str) -> None:
        from ai_pair_engineer.prompts import load_prompt

        assert load_prompt(name).strip()

    def test_unknown_prompt_name_raises(self) -> None:
        from ai_pair_engineer.prompts import load_prompt

        with pytest.raises(KeyError, match="unknown prompt"):
            load_prompt("nonexistent")

    def test_prompts_load_regardless_of_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The old code used a CWD-relative open() and broke outside the repo root."""
        from ai_pair_engineer.prompts import load_prompt

        monkeypatch.chdir(tmp_path)
        assert load_prompt("analyzer").strip()

    def test_prompts_specify_json_only_output(self) -> None:
        from ai_pair_engineer.prompts import load_prompt

        for name in ("analyzer", "tester", "refactor", "reviewer"):
            assert "no markdown fences" in load_prompt(name)


def _finding(
    *, category: Category = Category.ERROR_HANDLING, severity: Severity = Severity.HIGH
) -> object:
    from ai_pair_engineer.models.schemas import Finding

    titles = {
        Category.ERROR_HANDLING: "Unvalidated dict access raises KeyError",
        Category.COMPLEXITY: "Deeply nested conditionals",
        Category.READABILITY: "Inconsistent key naming",
        Category.SECURITY: "Shell injection via string concatenation",
    }
    return Finding(
        severity=severity,
        category=category,
        title=titles[category],
        description="d",
        recommendation="r",
    )


def _findings() -> list:
    return [
        _finding(category=Category.ERROR_HANDLING, severity=Severity.HIGH),
        _finding(category=Category.COMPLEXITY, severity=Severity.MEDIUM),
        _finding(category=Category.READABILITY, severity=Severity.LOW),
        _finding(category=Category.SECURITY, severity=Severity.CRITICAL),
    ]
