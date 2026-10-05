"""The analysis pipeline.

Four stages, each seeing only the context it needs:

    analyzer ──> tester  ──┐
        └─────> refactor ──┴──> reviewer

The test and refactor stages run off the same analyzer result but are not
sequential relative to each other. They could execute concurrently; they are
kept ordered here so token usage and failure reporting stay deterministic.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ai_pair_engineer.agents import (
    AnalyzerAgent,
    PipelineTrace,
    RefactorAgent,
    ReviewerAgent,
    StageUsage,
    TesterAgent,
)
from ai_pair_engineer.models.schemas import (
    AnalysisResult,
    FinalReview,
    RefactorResult,
    Severity,
    TestResult,
)
from ai_pair_engineer.static import StaticReport, analyze_source

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ("python", "javascript", "typescript", "java")


class UnsupportedLanguageError(ValueError):
    """The requested language is not supported."""


@dataclass
class PipelineResult:
    """Everything a single pipeline run produced."""

    language: str
    source_code: str
    static_report: StaticReport
    analysis: AnalysisResult
    tests: TestResult
    refactor: RefactorResult
    review: FinalReview
    trace: PipelineTrace = field(default_factory=PipelineTrace)

    @property
    def quality_score(self) -> int:
        return self.analysis.quality_score()

    @property
    def blocking_findings(self) -> list[Any]:
        return self.analysis.blocking()

    @property
    def highest_severity(self) -> Severity | None:
        findings = self.analysis.findings
        if not findings:
            return None
        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        present = {finding.severity for finding in findings}
        return next(severity for severity in order if severity in present)

    def to_dict(self, *, include_code: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "language": self.language,
            "summary": self.analysis.summary,
            "quality_score": self.quality_score,
            "severity_counts": {
                severity.value: count for severity, count in self.analysis.severity_counts().items()
            },
            "findings": [
                {
                    "severity": finding.severity.value,
                    "category": finding.category.value,
                    "title": finding.title,
                    "description": finding.description,
                    "recommendation": finding.recommendation,
                    "confidence": finding.confidence,
                    "line": finding.location.line if finding.location else None,
                    "symbol": finding.location.symbol if finding.location else None,
                }
                for finding in self.analysis.findings
            ],
            "tests": [
                {"name": test.name, "purpose": test.purpose, "covers": test.covers}
                for test in self.tests.tests
            ],
            "refactor": {
                "summary": self.refactor.summary,
                "changes": [change.model_dump() for change in self.refactor.changes],
                "risks": self.refactor.risks,
            },
            "review": self.review.model_dump(),
            "usage": self.trace.to_dict(),
        }
        if include_code:
            payload["refactored_code"] = self.refactor.refactored_code
        return payload


ProgressHook = Callable[[str, str], None]
"""Called as ``hook(stage_name, status)`` so a UI can show what is happening."""


def run_pipeline(
    language: str,
    source_code: str,
    *,
    model: str | None = None,
    on_progress: ProgressHook | None = None,
    static_report: StaticReport | None = None,
) -> PipelineResult:
    """Run all four stages and return the combined result.

    Args:
        language: one of :data:`SUPPORTED_LANGUAGES`, case-insensitive.
        source_code: the code to review. Treated as untrusted data.
        model: optional model override forwarded to every stage.
        on_progress: optional callback invoked as each stage starts and ends.
        static_report: precomputed local analysis, to avoid re-parsing.

    Raises:
        UnsupportedLanguageError: unknown language.
        ValueError: if ``source_code`` is empty.
        StageError: if any stage cannot produce a valid result.
    """
    normalized = _normalize_language(language)
    started = time.perf_counter()
    report = static_report or analyze_source(source_code, normalized)

    if report.syntax_error:
        logger.warning("Submitted source does not parse: %s", report.syntax_error)

    trace = PipelineTrace()

    analysis = _timed(
        AnalyzerAgent(normalized, source_code, report, model=model), trace, on_progress
    )
    tests = _timed(
        TesterAgent(normalized, source_code, analysis.findings, model=model),
        trace,
        on_progress,
    )
    refactor = _timed(
        RefactorAgent(normalized, source_code, analysis.findings, model=model),
        trace,
        on_progress,
    )
    review = _timed(
        ReviewerAgent(
            source_code,
            refactor.refactored_code,
            tests.tests,
            analysis.findings,
            model=model,
        ),
        trace,
        on_progress,
    )

    logger.info(
        "Pipeline finished in %.1fs using %d tokens across %d stages",
        time.perf_counter() - started,
        trace.total_tokens,
        len(trace.stages),
    )

    return PipelineResult(
        language=normalized,
        source_code=source_code,
        static_report=report,
        analysis=analysis,
        tests=tests,
        refactor=refactor,
        review=review,
        trace=trace,
    )


def _timed(agent: Any, trace: PipelineTrace, on_progress: ProgressHook | None) -> Any:
    """Run one stage, recording wall time and reporting progress."""
    if on_progress is not None:
        on_progress(agent.stage_name, "started")

    started = time.perf_counter()
    try:
        result = agent.run()
    finally:
        elapsed = time.perf_counter() - started
        if on_progress is not None:
            on_progress(agent.stage_name, f"finished in {elapsed:.1f}s")

    trace.record(StageUsage(stage=agent.stage_name, model=agent.model or "(default)"))
    return result


def _normalize_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(
            f"unsupported language {language!r}; expected one of {', '.join(SUPPORTED_LANGUAGES)}"
        )
    return normalized
