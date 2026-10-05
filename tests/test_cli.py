"""Tests for the command line interface.

Exit codes matter more than output formatting here: the CLI is meant to gate a
pipeline, so a wrong code silently turns a failing check into a green build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai_pair_engineer.cli import (
    EXIT_CONFIG,
    EXIT_FAILED,
    EXIT_FINDINGS,
    EXIT_OK,
    EXIT_USAGE,
    main,
)
from ai_pair_engineer.pipeline import UnsupportedLanguageError
from ai_pair_engineer.services.llm import LLMError, MissingAPIKeyError

SAMPLE = 'def add(a, b):\n    """Add."""\n    return a + b\n'


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


class TestArgumentHandling:
    def test_missing_path_is_a_usage_error(self) -> None:
        assert main([]) == EXIT_USAGE

    def test_nonexistent_file_is_a_usage_error(self, tmp_path: Path) -> None:
        assert main([str(tmp_path / "nope.py")]) == EXIT_USAGE

    def test_empty_file_is_rejected(self, tmp_path: Path, capsys: Any) -> None:
        path = tmp_path / "empty.py"
        path.write_text("   \n", encoding="utf-8")
        assert main([str(path)]) == EXIT_USAGE
        assert "no source code" in capsys.readouterr().err

    def test_reads_from_stdin(self, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
        import io
        import sys

        monkeypatch.setattr(sys, "stdin", io.StringIO(SAMPLE))
        assert main(["--stdin", "--static-only"]) == EXIT_OK
        assert "1 functions" in capsys.readouterr().out


class TestStaticOnly:
    def test_runs_without_an_api_key(self, sample_file: Path, capsys: Any) -> None:
        assert main([str(sample_file), "--static-only"]) == EXIT_OK
        assert "functions" in capsys.readouterr().out

    def test_reports_per_function_metrics(self, sample_file: Path, capsys: Any) -> None:
        main([str(sample_file), "--static-only"])
        assert "add" in capsys.readouterr().out

    def test_reports_a_mutable_default(self, tmp_path: Path, capsys: Any) -> None:
        path = tmp_path / "bad.py"
        path.write_text("def f(bucket=[]):\n    return bucket\n", encoding="utf-8")
        main([str(path), "--static-only"])
        assert "mutable default" in capsys.readouterr().out

    def test_reports_a_bare_except(self, tmp_path: Path, capsys: Any) -> None:
        path = tmp_path / "bad.py"
        path.write_text(
            "def f():\n    try:\n        return 1\n    except:\n        return 0\n",
            encoding="utf-8",
        )
        main([str(path), "--static-only"])
        assert "bare except" in capsys.readouterr().out


class TestPipelineInvocations:
    def test_missing_key_exits_with_the_config_code(self, sample_file: Path, capsys: Any) -> None:
        with patch(
            "ai_pair_engineer.services.llm.get_client",
            side_effect=MissingAPIKeyError("no key"),
        ):
            assert main([str(sample_file)]) == EXIT_CONFIG
        err = capsys.readouterr().err
        assert "no key" in err
        assert "--static-only" in err

    def test_language_failure_exits_with_the_usage_code(
        self, sample_file: Path, capsys: Any
    ) -> None:
        """An unsupported language is the caller's mistake, not a tool failure."""
        with patch(
            "ai_pair_engineer.agents.base.ask_llm",
            side_effect=UnsupportedLanguageError("bad language"),
        ):
            assert main([str(sample_file)]) == EXIT_USAGE
        assert "bad language" in capsys.readouterr().err

    def test_transport_failure_exits_with_the_failed_code(
        self, sample_file: Path, capsys: Any
    ) -> None:
        """An upstream outage is a failure, distinct from bad input or config."""
        with patch("ai_pair_engineer.agents.base.ask_llm", side_effect=LLMError("upstream 503")):
            assert main([str(sample_file)]) == EXIT_FAILED
        assert "upstream 503" in capsys.readouterr().err


class TestOutputFormats:
    @pytest.fixture(autouse=True)
    def _runnable(self, canned: dict[str, str], sample_file: Path) -> None:
        def dispatch(**call: Any) -> str:
            prompt = call["system_prompt"].lower()
            if "test engineer" in prompt:
                return canned["tests"]
            if "senior engineer improving" in prompt:
                return canned["refactor"]
            if "adversarial reviewer" in prompt:
                return canned["review"]
            return canned["analysis"]

        self._patcher = patch("ai_pair_engineer.agents.base.ask_llm", side_effect=dispatch)
        self._patcher.start()

    @pytest.fixture(autouse=True)
    def _stop(self) -> Any:
        yield
        self._patcher.stop()

    def test_json_output_parses(self, sample_file: Path, capsys: Any) -> None:
        main([str(sample_file), "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["language"] == "python"
        assert "findings" in payload

    def test_json_output_includes_the_refactored_code(self, sample_file: Path, capsys: Any) -> None:
        main([str(sample_file), "--format", "json"])
        assert "refactored_code" in json.loads(capsys.readouterr().out)

    def test_text_output_shows_the_verdict(self, sample_file: Path, capsys: Any) -> None:
        main([str(sample_file)])
        assert "VERDICT" in capsys.readouterr().out.upper()

    def test_text_output_shows_the_quality_score(self, sample_file: Path, capsys: Any) -> None:
        main([str(sample_file)])
        assert "Quality score" in capsys.readouterr().out

    def test_markdown_output_is_a_table(self, sample_file: Path, capsys: Any) -> None:
        main([str(sample_file), "--format", "markdown"])
        out = capsys.readouterr().out
        assert out.startswith("## AI Pair Engineer")
        assert "| Severity |" in out

    def test_markdown_output_lists_remaining_issues(self, sample_file: Path, capsys: Any) -> None:
        main([str(sample_file), "--format", "markdown"])
        assert "### Remaining issues" in capsys.readouterr().out

    def test_no_color_strips_escape_codes(self, sample_file: Path, capsys: Any) -> None:
        main([str(sample_file), "--no-color"])
        assert "\033[" not in capsys.readouterr().out


class TestExitCodes:
    @pytest.fixture(autouse=True)
    def _runnable(self, canned: dict[str, str]) -> None:
        def dispatch(**call: Any) -> str:
            prompt = call["system_prompt"].lower()
            if "test engineer" in prompt:
                return canned["tests"]
            if "senior engineer improving" in prompt:
                return canned["refactor"]
            if "adversarial reviewer" in prompt:
                return canned["review"]
            return canned["analysis"]

        self._patcher = patch("ai_pair_engineer.agents.base.ask_llm", side_effect=dispatch)
        self._patcher.start()

    @pytest.fixture(autouse=True)
    def _stop(self) -> Any:
        yield
        self._patcher.stop()

    def test_the_default_threshold_fails_on_a_high_finding(self, sample_file: Path) -> None:
        """Default is `high`, so a serious defect must break the build."""
        assert main([str(sample_file)]) == EXIT_FINDINGS

    def test_critical_threshold_passes_when_only_a_high_finding_exists(
        self, sample_file: Path
    ) -> None:
        assert main([str(sample_file), "--fail-on", "critical"]) == EXIT_OK

    def test_high_threshold_fails_on_a_high_finding(self, sample_file: Path) -> None:
        assert main([str(sample_file), "--fail-on", "high"]) == EXIT_FINDINGS

    def test_never_disables_failure(self, sample_file: Path) -> None:
        assert main([str(sample_file), "--fail-on", "never"]) == EXIT_OK

    def test_low_threshold_also_fails(self, sample_file: Path) -> None:
        assert main([str(sample_file), "--fail-on", "low"]) == EXIT_FINDINGS


class TestExitCodeLogic:
    def test_clean_code_passes(self) -> None:
        from ai_pair_engineer.cli import _exit_code
        from ai_pair_engineer.models.schemas import AnalysisResult

        result = _PipelineStub(AnalysisResult(summary="clean", findings=[]))
        assert _exit_code(result, "critical") == EXIT_OK

    def test_critical_fails_at_every_threshold(self) -> None:
        from ai_pair_engineer.cli import _exit_code
        from ai_pair_engineer.models.schemas import AnalysisResult, Category, Finding, Severity

        finding = Finding(
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            title="t",
            description="d",
            recommendation="r",
        )
        result = _PipelineStub(AnalysisResult(summary="s", findings=[finding]))
        for threshold in ("critical", "high", "medium", "low"):
            assert _exit_code(result, threshold) == EXIT_FINDINGS

    def test_never_always_passes(self) -> None:
        from ai_pair_engineer.cli import _exit_code
        from ai_pair_engineer.models.schemas import AnalysisResult, Category, Finding, Severity

        finding = Finding(
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            title="t",
            description="d",
            recommendation="r",
        )
        result = _PipelineStub(AnalysisResult(summary="s", findings=[finding]))
        assert _exit_code(result, "never") == EXIT_OK


class _PipelineStub:
    """Minimal stand-in exposing only what _exit_code reads."""

    def __init__(self, analysis: Any) -> None:
        self.analysis = analysis


class TestLanguageDetection:
    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            (".py", "python"),
            (".js", "javascript"),
            (".mjs", "javascript"),
            (".ts", "typescript"),
            (".tsx", "typescript"),
            (".java", "java"),
            (".txt", "python"),
        ],
    )
    def test_detects_language_from_the_suffix(
        self, tmp_path: Path, suffix: str, expected: str
    ) -> None:
        from ai_pair_engineer.cli import _detect_language

        path = tmp_path / f"module{suffix}"
        path.write_text("x = 1", encoding="utf-8")
        assert _detect_language(path, "x = 1", None) == expected

    def test_an_explicit_language_wins(self, tmp_path: Path) -> None:
        from ai_pair_engineer.cli import _detect_language

        path = tmp_path / "module.py"
        path.write_text("x = 1", encoding="utf-8")
        assert _detect_language(path, "x = 1", "java") == "java"

    def test_falls_back_to_python_without_a_path(self) -> None:
        from ai_pair_engineer.cli import _detect_language

        assert _detect_language(None, "x = 1", None) == "python"
