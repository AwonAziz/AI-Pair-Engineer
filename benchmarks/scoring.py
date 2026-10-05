"""Aggregate case results into tier scores.

The scoring rules, and why:

- **Recall** is the headline number: detected planted defects over planted
  defects. Recall is reported per tier and per category, because a tool that
  catches complexity problems and misses injection is not a working reviewer.
- **Groundedness** is a noise signal, not precision. See ``models.py`` for why
  the distinction matters and why it would be dishonest to call it precision.
- **Clean-case noise** is the one genuine false-positive measure, since a clean
  file has no true positives to hide behind.
- **Severity-weighted recall** weights a missed injection like a missed
  readability nit, because treating them equally would flatter the tool.

Macro and micro recall are both reported. Macro averages the per-case rate, so
a case with one planted defect counts as much as a case with six. Micro pools
every defect. A tier can look good on one and bad on the other, and reporting
only the flattering one is how a benchmark ends up misleading.
"""

from __future__ import annotations

from ai_pair_engineer.models.schemas import Category, Severity
from benchmarks.models import CaseResult, DetectionCase, Tier, TierScore

# Severity ordering, most serious first.
_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
)


def score_case(case: DetectionCase, result: CaseResult) -> CaseResult:
    """Fill in a case result's misses and unmatched findings.

    Split out from scoring so a caller can reuse it to inspect a single case
    without computing aggregate numbers.
    """
    detected = result.detected
    result.missed = [defect.id for defect in case.defects if defect.id not in detected]
    return result


def recall_by_case(results: list[tuple[DetectionCase, CaseResult]]) -> float | None:
    """Macro recall: the mean per-case recall across cases that had defects."""
    rates = [
        len(result.detected) / len(case.defects)
        for case, result in results
        if case.defects and result.error is None
    ]
    if not rates:
        return None
    return sum(rates) / len(rates)


def recall_by_category(
    results: list[tuple[DetectionCase, CaseResult]],
) -> dict[Category, float | None]:
    """Macro recall per planted-defect category.

    A category with no planted defects in the corpus reports ``None`` rather
    than a flattering 1.0.
    """
    buckets: dict[Category, list[float]] = {}

    for case, result in results:
        if case.defects or result.error is not None:
            detected = result.detected
            for defect in case.defects:
                rate = 1.0 if defect.id in detected else 0.0
                buckets.setdefault(defect.category, []).append(rate)

    return {
        category: (sum(rates) / len(rates) if rates else None)
        for category, rates in sorted(buckets.items(), key=lambda item: item[0].value)
    }


def recall_by_severity(
    results: list[tuple[DetectionCase, CaseResult]],
) -> dict[Severity, float | None]:
    """Macro recall per planted-defect severity, ordered most serious first."""
    buckets: dict[Severity, list[float]] = {}

    for case, result in results:
        if result.error is not None:
            continue
        detected = result.detected
        for defect in case.defects:
            rate = 1.0 if defect.id in detected else 0.0
            buckets.setdefault(defect.severity, []).append(rate)

    return {
        severity: (sum(buckets[severity]) / len(buckets[severity]) if severity in buckets else None)
        for severity in _SEVERITY_ORDER
    }


def severity_weighted_recall(
    results: list[tuple[DetectionCase, CaseResult]],
) -> float | None:
    """Recall weighted by severity.

    A missed ``critical`` counts four times a missed ``low``. The weights are a
    judgement call, stated here so the number can be argued with rather than
    taken on faith.
    """
    weights = {
        Severity.CRITICAL: 4.0,
        Severity.HIGH: 3.0,
        Severity.MEDIUM: 2.0,
        Severity.LOW: 1.0,
    }

    earned = 0.0
    possible = 0.0
    for case, result in results:
        if result.error is not None:
            continue
        detected = result.detected
        for defect in case.defects:
            weight = weights[defect.severity]
            possible += weight
            if defect.id in detected:
                earned += weight

    return earned / possible if possible else None


def aggregate(
    tier: Tier,
    results: list[tuple[DetectionCase, CaseResult]],
    *,
    with_findings: list[CaseResult] | None = None,
) -> TierScore:
    """Build one tier's aggregate score.

    ``with_findings`` carries the findings to count as noise, which differs from
    the cases scored for recall: a clean case contributes noise but no recall.
    """
    scored = [(case, result) for case, result in results if result.error is None]
    finding_source = with_findings if with_findings is not None else [r for _, r in results]

    planted = sum(len(case.defects) for case, _ in scored)
    detected = sum(len(result.detected) for _, result in scored)

    clean_results = [result for case, result in results if case.clean and result.error is None]

    return TierScore(
        tier=tier,
        cases=len(results),
        clean_cases=len(clean_results),
        planted=planted,
        detected=detected,
        findings=sum(len(result.findings) for result in finding_source),
        matched_findings=sum(len(result.matches) for result in finding_source),
        clean_case_findings=sum(len(result.findings) for result in clean_results),
    )
