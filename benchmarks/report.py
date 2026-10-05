"""Render a run report.

Two formats: Markdown for a pull request or a README, JSON for programmatic
comparison across runs.

The Markdown report leads with the table that matters and states what the
numbers do and do not mean. A benchmark whose output invites over-reading is
worse than no benchmark, so the caveats travel with the figures rather than
living in a separate section nobody reads.

Numbers are printed as percentages to one decimal. Exact counts appear in the
per-case detail, because "recall 0.62" invites a precision that the corpus
does not support: with a corpus this size the confidence interval on a recall
figure is several points wide.
"""

from __future__ import annotations

import json
from typing import Any

from ai_pair_engineer.models.schemas import Category, Severity
from benchmarks.models import RunReport, Tier

# Placeholder rendered where a metric does not apply, e.g. recall on a corpus
# with no planted defects. Plain ASCII so Windows consoles do not render it as
# mojibake.
NOT_APPLICABLE = "n/a"


def _percent(value: float | None) -> str:
    if value is None:
        return NOT_APPLICABLE
    return f"{value * 100:.1f}%"


def _usd(value: float) -> str:
    if value == 0:
        return "not priced"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def render_markdown(report: RunReport, *, verbose: bool = False) -> str:
    """Render the full Markdown report."""
    lines: list[str] = ["# AI Pair Engineer benchmark", ""]

    lines += [
        f"- **Model:** `{report.model}`",
        f"- **Temperature:** {report.temperature}",
        f"- **Corpus:** {report.corpus_size} file(s)",
        f"- **Cost:** {_usd(report.total_cost_usd)} "
        f"across {report.total_calls} model call(s), {report.total_tokens} tokens",
        "",
    ]

    lines += _render_detection_table(report)

    if report.regression_results:
        lines += _render_regression_section(report)

    if verbose:
        lines += _render_detail(report)

    lines += _render_caveats(report)
    return "\n".join(lines) + "\n"


def _render_detection_table(report: RunReport) -> list[str]:
    if not report.scores:
        return []

    lines = ["## Defect detection", ""]
    lines += [
        "| Tier | Recall | Groundedness | Findings | Clean-file noise |",
        "|---|---|---|---|---|",
    ]

    for score in report.scores:
        lines.append(
            f"| {score.tier.label} "
            f"| {_percent(score.recall)} "
            f"| {_percent(score.groundedness)} "
            f"| {score.findings} "
            f"| {_percent(score.clean_noise)} |"
        )

    lines += [
        "",
        "**Recall** is the headline: of the defects planted in the corpus, how "
        "many did the tier report? **Groundedness** is the share of findings that "
        "match a planted defect, and is a noise signal rather than a precision "
        "claim. **Clean-file noise** is findings on files with no planted "
        "defects, which is the only genuine false-positive measure here.",
        "",
    ]

    return lines


def _render_regression_section(report: RunReport) -> list[str]:
    results = report.regression_results
    caught = sum(1 for result in results if result.caught)
    approved = sum(1 for result in results if result.approved)
    mentioned = sum(1 for result in results if result.mentioned)
    total = len(results)

    lines = [
        "## Reviewer behaviour-change detection",
        "",
        f"Caught **{caught}/{total}**. Approved anyway in {approved}/{total}, "
        f"noticed something but did not name it in {mentioned}/{total}.",
        "",
        "| Case | Caught | Verdict | Score | What it should have caught |",
        "|---|---|---|---|---|",
    ]

    for result in results:
        mark = "yes" if result.caught else "no"
        verdict = result.verdict if result.error is None else "error"
        lines.append(
            f"| {result.case_id} | {mark} | {verdict} | {result.score} "
            f"| {_first_line(result.reasoning)} |"
        )

    lines += [
        "",
        "A case counts as caught only when the reviewer rejected the refactor "
        "*and* named the specific behaviour change. Rejecting for vague reasons, "
        "or approving while noticing something, both score as misses.",
        "",
    ]
    return lines


def _render_detail(report: RunReport) -> list[str]:
    lines = ["## Per-case detail", ""]

    for result in report.case_results:
        lines += [f"### {result.case_id} ({result.tier.label})", ""]

        if result.error:
            lines += [f"Error: `{result.error}`", ""]
            continue

        if result.matches:
            lines += ["Detected:", ""]
            for match in result.matches:
                lines.append(f"- `{match.defect_id}` via {match.rule.value}: {match.finding_title}")
            lines.append("")

        if result.missed:
            lines += ["Missed: " + ", ".join(f"`{m}`" for m in result.missed), ""]

        if result.unmatched:
            lines += ["Unmatched findings:", ""]
            for finding in result.unmatched:
                lines.append(f"- {finding.title}")
            lines.append("")

    return lines


def _render_caveats(report: RunReport) -> list[str]:
    """State the limits of the numbers, in the same document as the numbers."""
    planted = sum(score.planted for score in report.scores)
    unique_planted = planted // max(1, len(report.scores))

    return [
        "## What these numbers do not say",
        "",
        f"- The corpus holds {unique_planted} planted defect(s) across "
        f"{report.corpus_size} file(s). That is small enough that the confidence "
        "interval on any recall figure spans several percentage points. Treat "
        "differences under about ten points as noise.",
        "- Groundedness is not precision. A finding that matches no planted "
        "defect may be a real problem the corpus did not annotate. Counting "
        "those as errors would measure the annotations, not the tool.",
        "- The corpus is Python only, hand-written, and built by the same person "
        "who wrote the analyzer. That is a real conflict of interest. An "
        "independent corpus is the obvious next step.",
        "- Static analysis is reported separately because it is deterministic and "
        "free. It is a lower bound on what is catchable without a model, not a "
        "competitor to the model tiers.",
        "- Cost figures come from a published price table and drift. Token counts "
        "are the measured quantity; dollars are an estimate.",
    ]


def _first_line(text: str, limit: int = 90) -> str:
    """First line of ``text``, truncated for a table cell."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed or NOT_APPLICABLE
    return collapsed[: limit - 1].rstrip() + "..."


def render_json(report: RunReport, *, verbose: bool = False) -> str:
    """Render the machine-readable report."""
    payload: dict[str, Any] = report.to_dict()
    if not verbose:
        payload.pop("case_results", None)
    return json.dumps(payload, indent=2)


def render_category_breakdown(
    report: RunReport, category_recall: dict[Category, float | None]
) -> list[str]:
    """Per-category recall table, for spotting a blind spot."""
    lines = ["| Category | " + " | ".join(score.tier.label for score in report.scores) + " |"]
    lines.append("|---" * (len(report.scores) + 1) + "|")

    for category in Category:
        lines.append(
            f"| {category.value} | "
            + " | ".join(_percent(category_recall.get(category)) for _ in report.scores)
            + " |"
        )

    return lines


def render_severity_breakdown(
    report: RunReport, severity_recall: dict[Severity, float | None]
) -> list[str]:
    """Per-severity recall table."""
    lines = ["| Severity | " + " | ".join(score.tier.label for score in report.scores) + " |"]
    lines.append("|---" * (len(report.scores) + 1) + "|")

    for severity in (
        Severity.CRITICAL,
        Severity.HIGH,
        Severity.MEDIUM,
        Severity.LOW,
    ):
        lines.append(
            f"| {severity.value} | "
            + " | ".join(_percent(severity_recall.get(severity)) for _ in report.scores)
            + " |"
        )

    return lines


def tier_order(tiers: list[str]) -> list[Tier]:
    """Parse tier names from the command line, preserving benchmark order."""
    ordered = [Tier.STATIC, Tier.NAIVE, Tier.ANALYZER, Tier.ANALYZER_STATIC]
    if "all" in tiers:
        return ordered
    return [tier for tier in ordered if tier.value in tiers]
