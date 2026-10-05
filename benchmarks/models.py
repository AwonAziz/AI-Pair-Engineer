"""Benchmark data model.

The corpus is a set of files with a known set of planted defects. A benchmark
run asks a tier to review each file and then scores what came back against
ground truth.

Scoring terminology, since these words get used loosely:

- **Recall**: of the defects that were planted, how many did the tier report?
  This is the number that matters most, because a missed security defect is
  worse than a noisy review.
- **Groundedness**: of the findings the tier produced, what share correspond to
  a real planted defect? This is *not* precision. An unrecognised finding may
  be a true positive that was simply not in the ground truth, and a benchmark
  that counted those as false would be measuring its own annotation, not the
  tool. It is reported as a signal for noise, never as a precision claim.
- **Clean-case noise**: findings produced on files that contain no planted
  defects. Here a finding really is very likely a false positive, so this is
  the one place false-positive counting is meaningful.

That distinction is the reason a benchmark is worth building at all. It is easy
to publish a number that flatters the tool by quietly counting true positives
as errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ai_pair_engineer.models.schemas import Category, Finding, Severity


class MatchRule(StrEnum):
    """How a finding was tied to a planted defect."""

    #: The finding pointed at the defect's line, or inside its line span.
    LOCATED = "located"
    #: The finding's text matched the defect's keyword groups.
    SEMANTIC = "semantic"
    #: Both rules fired, which is the strongest possible evidence.
    BOTH = "both"


class Tier(StrEnum):
    """A way of reviewing a file, for comparison."""

    #: Local AST analysis only. No model calls, no cost.
    STATIC = "static"
    #: One model call, no system prompt, no structure. The baseline.
    NAIVE = "naive"
    #: The real analyzer stage, without local static evidence.
    ANALYZER = "analyzer"
    #: The real analyzer stage, with local static evidence.
    ANALYZER_STATIC = "analyzer+static"

    @property
    def uses_model(self) -> bool:
        return self is not Tier.STATIC

    @property
    def label(self) -> str:
        return _TIER_LABELS[self]


_TIER_LABELS: dict[Tier, str] = {
    Tier.STATIC: "static only",
    Tier.NAIVE: "naive (1 call)",
    Tier.ANALYZER: "analyzer",
    Tier.ANALYZER_STATIC: "analyzer + static",
}

#: Line tolerance for :attr:`MatchRule.LOCATED`. One line of slack absorbs
#: off-by-one reporting without letting a finding match a distant defect.
LINE_TOLERANCE = 1


@dataclass(frozen=True)
class PlantedDefect:
    """A defect deliberately written into a corpus file.

    ``keywords`` are groups, not a flat list. Every group must match somewhere
    in the finding's text, which makes a match specific: a defect described as
    ``[["bare", "empty"], ["except"]]`` needs both an except-ish term and a
    bare-or-empty term, so a finding that merely mentions exception handling
    does not count.
    """

    id: str
    summary: str
    category: Category
    severity: Severity
    line: int
    end_line: int | None = None
    keywords: tuple[tuple[str, ...], ...] = ()
    note: str = ""

    @property
    def span(self) -> tuple[int, int]:
        return (self.line, self.end_line or self.line)

    def contains_line(self, line: int) -> bool:
        start, end = self.span
        return start - LINE_TOLERANCE <= line <= end + LINE_TOLERANCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "category": self.category.value,
            "severity": self.severity.value,
            "line": self.line,
            "end_line": self.end_line,
            "note": self.note,
        }


@dataclass(frozen=True)
class DetectionCase:
    """One corpus file and the defects planted in it."""

    id: str
    language: str
    source: str
    defects: tuple[PlantedDefect, ...] = ()
    intent: str = ""
    tags: tuple[str, ...] = ()
    #: True when the file is meant to be clean. Any finding on a clean case is
    #: counted as noise rather than as recall.
    clean: bool = False

    def defect_by_id(self, defect_id: str) -> PlantedDefect | None:
        return next((d for d in self.defects if d.id == defect_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language,
            "clean": self.clean,
            "intent": self.intent,
            "tags": list(self.tags),
            "defects": [defect.to_dict() for defect in self.defects],
        }


@dataclass(frozen=True)
class Match:
    """A finding tied to a defect."""

    defect_id: str
    finding_title: str
    rule: MatchRule

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_id": self.defect_id,
            "finding_title": self.finding_title,
            "rule": self.rule.value,
        }


@dataclass
class CaseResult:
    """What one tier produced for one case."""

    case_id: str
    tier: Tier
    findings: list[Finding] = field(default_factory=list)
    matches: list[Match] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    #: Findings that matched no planted defect. On a clean case every finding
    #: lands here, which is what makes clean-case noise measurable.
    unmatched: list[Finding] = field(default_factory=list)
    error: str | None = None

    @property
    def detected(self) -> set[str]:
        return {match.defect_id for match in self.matches}

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tier": self.tier.value,
            "findings": [
                {
                    "title": finding.title,
                    "severity": finding.severity.value,
                    "category": finding.category.value,
                    "line": finding.location.line if finding.location else None,
                }
                for finding in self.findings
            ],
            "matches": [match.to_dict() for match in self.matches],
            "missed": self.missed,
            "unmatched": [finding.title for finding in self.unmatched],
            "error": self.error,
        }


@dataclass(frozen=True)
class TierScore:
    """Aggregate performance of one tier across the corpus."""

    tier: Tier
    cases: int = 0
    clean_cases: int = 0
    planted: int = 0
    detected: int = 0
    findings: int = 0
    matched_findings: int = 0
    clean_case_findings: int = 0

    @property
    def recall(self) -> float | None:
        if not self.planted:
            return None
        return self.detected / self.planted

    @property
    def groundedness(self) -> float | None:
        if not self.findings:
            return None
        return self.matched_findings / self.findings

    @property
    def clean_noise(self) -> float | None:
        if not self.clean_cases:
            return None
        return self.clean_case_findings / self.clean_cases

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "label": self.tier.label,
            "cases": self.cases,
            "clean_cases": self.clean_cases,
            "planted": self.planted,
            "detected": self.detected,
            "recall": self.recall,
            "findings": self.findings,
            "matched_findings": self.matched_findings,
            "groundedness": self.groundedness,
            "clean_case_findings": self.clean_case_findings,
            "clean_noise": self.clean_noise,
        }


@dataclass(frozen=True)
class RegressionCase:
    """A behaviour change a reviewer is supposed to notice.

    ``original`` and ``refactored`` differ by exactly one behavioural change.
    ``expected`` describes it in the reviewer's own vocabulary; the regression
    tier counts a case as caught only when the reviewer fails to approve *and*
    mentions at least one of those phrases.
    """

    id: str
    original: str
    refactored: str
    change: str
    expected: tuple[str, ...]
    language: str = "python"
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "change": self.change,
            "expected": list(self.expected),
            "language": self.language,
            "tags": list(self.tags),
        }


@dataclass
class RegressionResult:
    """Whether a reviewer caught one planted regression."""

    case_id: str
    tier: Tier
    approved: bool
    mentioned: bool
    verdict: str
    score: int
    reasoning: str
    cost_usd: float = 0.0
    tokens: int = 0
    error: str | None = None

    @property
    def caught(self) -> bool:
        """Caught means rejected *and* the reason was named.

        Requiring both is deliberate. A reviewer that rejects a correct
        refactor for vague reasons is not useful, and a reviewer that notices
        something is wrong but cannot say what has failed to act on it.
        """
        return not self.approved and self.mentioned

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tier": self.tier.value,
            "approved": self.approved,
            "mentioned": self.mentioned,
            "caught": self.caught,
            "verdict": self.verdict,
            "score": self.score,
            "reasoning": self.reasoning,
            "cost_usd": round(self.cost_usd, 6),
            "tokens": self.tokens,
            "error": self.error,
        }


@dataclass
class RunReport:
    """Everything one benchmark invocation produced."""

    model: str
    temperature: float
    scores: list[TierScore] = field(default_factory=list)
    case_results: list[CaseResult] = field(default_factory=list)
    regression_results: list[RegressionResult] = field(default_factory=list)
    corpus_size: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "corpus_size": self.corpus_size,
            "scores": [score.to_dict() for score in self.scores],
            "case_results": [result.to_dict() for result in self.case_results],
            "regression_results": [r.to_dict() for r in self.regression_results],
            "totals": {
                "tokens": self.total_tokens,
                "cost_usd": round(self.total_cost_usd, 6),
                "calls": self.total_calls,
            },
        }
