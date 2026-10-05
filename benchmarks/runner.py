"""Run the benchmark.

A run walks the corpus, asks each requested tier to review every file, and
scores the results.

Caching matters here. A full corpus run is several hundred model calls, so
re-running to check a code change would cost real money every time. Each
(case, tier, model, temperature, corpus-hash) result is cached on disk and
reused unless ``--refresh`` is passed. The cache key includes a hash of the
case source, so editing a corpus file invalidates only that case.

The harness records failures per case rather than aborting. A tier that
crashes on one file should still produce numbers for the rest, with the
failure visible in the report instead of silently shortening the corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_pair_engineer.agents import AnalyzerAgent, StageError
from ai_pair_engineer.models.schemas import (
    AnalysisResult,
    Category,
    Finding,
    Location,
    Severity,
)
from ai_pair_engineer.services.llm import (
    LLMError,
    TokenUsage,
    ask_llm,
    estimate_cost_usd,
    parse_structured_response,
)
from ai_pair_engineer.static import StaticReport, analyze_source
from benchmarks.corpus import load_cases, load_regressions
from benchmarks.matching import match_findings
from benchmarks.models import (
    CaseResult,
    DetectionCase,
    RegressionCase,
    RegressionResult,
    RunReport,
    Tier,
)
from benchmarks.naive import naive_analyzer_prompt
from benchmarks.scoring import aggregate, score_case

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Static analysis reports these with a synthetic location so the same scoring
# path applies to the no-model tier. Each report becomes one finding.
_STATIC_CATEGORY_BY_SIGNAL: dict[str, tuple[Category, str]] = {
    "bare_except": (Category.ERROR_HANDLING, "Bare except swallows every error"),
    "mutable_default": (
        Category.ERROR_HANDLING,
        "Mutable default argument is shared across calls",
    ),
}


@dataclass
class TierRun:
    """Accounting for one tier's pass over the corpus."""

    tier: Tier
    tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0

    def record(self, usage: TokenUsage) -> None:
        self.calls += 1
        self.tokens += usage.total_tokens
        self.cost_usd += estimate_cost_usd(usage)


def _fingerprint(case: DetectionCase, tier: Tier, model: str, temperature: float) -> str:
    """Cache key for one case/tier combination."""
    material = "|".join(
        [
            case.id,
            case.source,
            ",".join(sorted(defect.id for defect in case.defects)),
            tier.value,
            model,
            f"{temperature}",
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str, tier: Tier) -> Path:
    return CACHE_DIR / f"{tier.value.replace('+', '_')}-{key}.json"


def _load_cache(path: Path) -> AnalysisResult | None:
    if not path.is_file():
        return None
    try:
        return AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def _store_cache(path: Path, result: AnalysisResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def review_with_static(case: DetectionCase) -> AnalysisResult:
    """Turn the local AST report into findings, without any model call.

    The no-model tier has to produce findings in the same schema as every other
    tier, so deterministic signals are reported as synthetic findings. Line
    numbers are exact, which is why this tier is worth measuring separately: it
    catches what it catches without any guessing at all.
    """
    report = analyze_source(case.source, case.language)
    findings: list[Finding] = []

    for line in report.bare_except_lines:
        category, title = _STATIC_CATEGORY_BY_SIGNAL["bare_except"]
        findings.append(_static_finding(category, title, line, "Catch a specific exception type."))

    for line in report.mutable_default_lines:
        category, title = _STATIC_CATEGORY_BY_SIGNAL["mutable_default"]
        findings.append(
            _static_finding(category, title, line, "Use None as the default and create inside.")
        )

    return AnalysisResult(
        summary=f"Local static analysis found {len(findings)} issue(s).",
        findings=findings,
    )


def _static_finding(category: Category, title: str, line: int, recommendation: str) -> Finding:
    return Finding(
        severity=Severity.HIGH,
        category=category,
        title=title,
        description=f"Detected at line {line} by local AST analysis.",
        recommendation=recommendation,
        confidence=1.0,
        location=Location(line=line),
    )


def review_with_naive(
    case: DetectionCase, *, model: str, temperature: float, run: TierRun
) -> AnalysisResult:
    """Review with a single unstructured call: the thing this project replaces."""
    recorded: list[TokenUsage] = []
    raw = ask_naive(
        user_prompt=naive_analyzer_prompt(case.source, case.language),
        model=model,
        temperature=temperature,
        on_usage=recorded.append,
    )
    for usage in recorded:
        run.record(usage)
    return parse_structured_response(raw, AnalysisResult)


def ask_naive(**kwargs: Any) -> str:
    """Single seam for every model call this harness makes directly.

    The analyzer tiers call through ``agents.base.ask_llm`` instead, so tests
    that want to intercept all of them need both seams. Keeping them named and
    documented here is better than hiding one behind indirection that the
    analyzer path cannot reach.
    """
    return ask_llm(system_prompt="", **kwargs)


def review_with_analyzer(
    case: DetectionCase,
    *,
    tier: Tier,
    model: str,
    temperature: float,
    run: TierRun,
) -> AnalysisResult:
    """Review with the project's real analyzer stage."""
    agent = AnalyzerAgent(
        case.language,
        case.source,
        model=model,
        temperature=temperature,
        include_static=tier is Tier.ANALYZER_STATIC,
    )
    result = agent.run()
    run.calls += agent.usage.attempts
    run.tokens += agent.usage.total_tokens
    run.cost_usd += agent.usage.cost_usd
    return result


def review_case(
    case: DetectionCase,
    tier: Tier,
    *,
    model: str,
    temperature: float,
    run: TierRun,
    refresh: bool,
    use_cache: bool,
) -> CaseResult:
    """Review one case with one tier, using and updating the cache."""
    key = _fingerprint(case, tier, model, temperature)
    path = _cache_path(key, tier)

    if use_cache and not refresh:
        cached = _load_cache(path)
        if cached is not None:
            matches, unmatched = match_findings(cached.findings, case.defects)
            result = CaseResult(
                case_id=case.id,
                tier=tier,
                findings=cached.findings,
                matches=matches,
                unmatched=unmatched,
            )
            return score_case(case, result)

    try:
        started = time.perf_counter()
        if tier is Tier.STATIC:
            analysis = review_with_static(case)
        elif tier is Tier.NAIVE:
            analysis = review_with_naive(case, model=model, temperature=temperature, run=run)
        else:
            analysis = review_with_analyzer(
                case, tier=tier, model=model, temperature=temperature, run=run
            )
        run.seconds += time.perf_counter() - started

        if use_cache:
            _store_cache(path, analysis)

    except (LLMError, StageError, ValueError) as exc:
        # One case failing must not end the run. Record it and carry on, so the
        # report shows a short corpus rather than hiding the failure.
        #
        # StageError is caught explicitly because it subclasses RuntimeError, not
        # LLMError: a stage that cannot produce a valid result is the single most
        # likely failure here, and letting it propagate would abandon every case
        # after the first bad response.
        logger.warning("tier %s failed on case %s: %s", tier, case.id, exc)
        return CaseResult(case_id=case.id, tier=tier, error=str(exc))

    matches, unmatched = match_findings(analysis.findings, case.defects)
    return score_case(
        case,
        CaseResult(
            case_id=case.id,
            tier=tier,
            findings=analysis.findings,
            matches=matches,
            unmatched=unmatched,
        ),
    )


def run_regression_case(
    case: RegressionCase,
    *,
    tier: Tier,
    model: str,
    temperature: float,
    naive_reviewer: bool,
) -> RegressionResult:
    """Check whether a reviewer notices one planted behaviour change.

    ``naive_reviewer`` swaps the adversarial reviewer for a plain "compare these
    two versions and say if the refactor is safe" prompt. That comparison is the
    whole claim of this project, so it gets measured rather than asserted.
    """
    from benchmarks.reviewer_prompts import naive_reviewer_prompt

    recorded: list[TokenUsage] = []
    try:
        if naive_reviewer:
            user_prompt = naive_reviewer_prompt(case.original, case.refactored)
            raw = ask_naive(
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                on_usage=recorded.append,
            )
            from ai_pair_engineer.models.schemas import FinalReview

            review = parse_structured_response(raw, FinalReview)
        else:
            from ai_pair_engineer.agents import ReviewerAgent

            agent = ReviewerAgent(
                case.original,
                case.refactored,
                model=model,
                temperature=temperature,
            )
            review = agent.run()
            recorded.append(
                TokenUsage(
                    model=model,
                    prompt_tokens=agent.usage.input_tokens,
                    completion_tokens=agent.usage.output_tokens,
                )
            )

    except (LLMError, StageError, ValueError) as exc:
        return RegressionResult(
            case_id=case.id,
            tier=tier,
            approved=True,
            mentioned=False,
            verdict="error",
            score=0,
            reasoning=str(exc),
            error=str(exc),
        )

    # The reviewer's own words: summary, recommendation, and any remaining
    # issues. A finding can hide anywhere in there.
    haystack = " ".join(
        [
            review.summary,
            review.recommendation,
            *review.remaining_issues,
            *review.strengths,
        ]
    ).lower()

    mentioned = any(phrase.lower() in haystack for phrase in case.expected)

    tokens = sum(usage.total_tokens for usage in recorded)
    return RegressionResult(
        case_id=case.id,
        tier=tier,
        approved=review.approved,
        mentioned=mentioned,
        verdict=review.verdict,
        score=review.score,
        reasoning=review.summary or review.recommendation,
        cost_usd=sum(estimate_cost_usd(usage) for usage in recorded),
        tokens=tokens,
    )


def default_model() -> str:
    return os.getenv("MODEL", "deepseek/deepseek-chat").strip() or "deepseek/deepseek-chat"


def run_benchmark(
    tiers: list[Tier],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    refresh: bool = False,
    use_cache: bool = True,
    run_regressions: bool = False,
    naive_reviewer: bool = False,
    case_filter: list[str] | None = None,
    verbose: bool = False,
) -> RunReport:
    """Run every requested tier over the corpus and score the results.

    Args:
        tiers: which review strategies to compare.
        model: model id; defaults to ``$MODEL``.
        temperature: passed to every model call. Defaults to 0 so two runs of
            the same code are comparable.
        refresh: ignore and overwrite cached results.
        use_cache: reuse cached results. Only the static tier ignores the cache,
            since it is free and deterministic.
        run_regressions: also run the reviewer comparison.
        naive_reviewer: use the plain reviewer prompt for the regression tier.
        case_filter: restrict to these case ids.
        verbose: log each case as it completes.
    """
    resolved_model = model or default_model()
    cases = load_cases()
    if case_filter:
        wanted = set(case_filter)
        cases = [case for case in cases if case.id in wanted]
        if not cases:
            raise ValueError(f"no cases matched {sorted(wanted)}")

    report = RunReport(
        model=resolved_model,
        temperature=temperature,
        corpus_size=len(cases),
    )

    for tier in tiers:
        run = TierRun(tier=tier)
        results: list[tuple[DetectionCase, CaseResult]] = []

        for case in cases:
            outcome = review_case(
                case,
                tier,
                model=resolved_model,
                temperature=temperature,
                run=run,
                refresh=refresh,
                # The static tier is deterministic and free; caching it would
                # only risk serving a stale result.
                use_cache=use_cache and tier is not Tier.STATIC,
            )
            results.append((case, outcome))
            report.case_results.append(outcome)

            if verbose:
                status = "err" if outcome.error else f"{len(outcome.detected)} found"
                print(f"  [{tier.label}] {case.id}: {status}")

        report.scores.append(aggregate(tier, results, with_findings=[r for _, r in results]))
        report.total_tokens += run.tokens
        report.total_cost_usd += run.cost_usd
        report.total_calls += run.calls

    if run_regressions:
        regressions = load_regressions()
        if case_filter:
            wanted = set(case_filter)
            regressions = [r for r in regressions if r.id in wanted]

        for regression in regressions:
            result = run_regression_case(
                regression,
                tier=tiers[-1] if tiers else Tier.ANALYZER_STATIC,
                model=resolved_model,
                temperature=temperature,
                naive_reviewer=naive_reviewer,
            )
            report.regression_results.append(result)
            report.total_tokens += result.tokens
            report.total_cost_usd += result.cost_usd
            report.total_calls += 1 if result.error is None else 0

    return report


def build_static_report(source: str, language: str) -> StaticReport:
    """Expose local analysis for callers that want the raw report."""
    return analyze_source(source, language)
