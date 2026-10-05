"""Structured output contracts for every pipeline stage.

Every LLM stage is validated against one of these models. Anything that does
not conform is rejected rather than silently coerced, because a confidently
wrong review is worse than no review.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Severity weights used for the deterministic quality score. Keeping them in
# one place means the UI, the CLI, and the tests can never disagree.
SEVERITY_WEIGHTS: dict[Severity, int] = {
    Severity.CRITICAL: 25,
    Severity.HIGH: 15,
    Severity.MEDIUM: 8,
    Severity.LOW: 3,
}


class Category(StrEnum):
    # A correctness defect is one that produces the wrong answer or an unwanted
    # side effect. It is kept separate from error_handling because the two call
    # for different responses: a correctness defect needs a logic fix, an
    # error_handling defect needs a failure path. Conflating them is what makes
    # a reviewer's findings hard to act on.
    CORRECTNESS = "correctness"
    CODE_SMELL = "code_smell"
    MAINTAINABILITY = "maintainability"
    READABILITY = "readability"
    ARCHITECTURE = "architecture"
    COMPLEXITY = "complexity"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DUPLICATION = "duplication"


# Categories that describe "the code is hard to reason about". These drive the
# refactoring stage.
REFACTOR_CATEGORIES: frozenset[Category] = frozenset(
    {
        Category.CODE_SMELL,
        Category.MAINTAINABILITY,
        Category.READABILITY,
        Category.ARCHITECTURE,
        Category.COMPLEXITY,
        Category.DUPLICATION,
    }
)

# Categories that describe "the code is wrong, crashes, or is unsafe". These
# drive the test generation stage: a test exists to pin behaviour that is at
# risk of changing silently.
TEST_CATEGORIES: frozenset[Category] = frozenset(
    {
        Category.CORRECTNESS,
        Category.ERROR_HANDLING,
        Category.SECURITY,
    }
)


class Location(BaseModel):
    """Where a finding lives in the submitted source.

    Line numbers are optional because some findings span a whole file or
    describe an absence (for example, "no error handling anywhere").
    """

    line: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed line where the issue starts, if applicable.",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed line where the issue ends, if applicable.",
    )
    symbol: str | None = Field(
        default=None,
        description="Enclosing function, method, or class name, if known.",
    )

    def render(self) -> str:
        parts: list[str] = []
        if self.line is not None:
            if self.end_line is not None and self.end_line != self.line:
                parts.append(f"lines {self.line}-{self.end_line}")
            else:
                parts.append(f"line {self.line}")
        if self.symbol:
            parts.append(f"in `{self.symbol}`")
        return ", ".join(parts)


class Finding(BaseModel):
    severity: Severity
    category: Category
    title: str
    description: str
    recommendation: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    location: Location | None = None

    @field_validator("title", "description", "recommendation")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @property
    def weight(self) -> int:
        return SEVERITY_WEIGHTS[self.severity]


class AnalysisResult(BaseModel):
    summary: str
    findings: list[Finding] = Field(default_factory=list)

    def severity_counts(self) -> dict[Severity, int]:
        counts: dict[Severity, int] = dict.fromkeys(Severity, 0)
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    def quality_score(self) -> int:
        """Deterministic 0-100 score derived from finding severities."""
        deduction = sum(finding.weight for finding in self.findings)
        return max(0, 100 - deduction)

    def blocking(self) -> list[Finding]:
        """Findings severe enough to justify failing a pipeline run."""
        return [
            finding
            for finding in self.findings
            if finding.severity in {Severity.CRITICAL, Severity.HIGH}
        ]


class TestCase(BaseModel):
    name: str
    purpose: str
    test_code: str
    covers: list[str] = Field(
        default_factory=list,
        description="Titles of findings this test is intended to cover.",
    )

    @field_validator("test_code")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("test_code must not be empty")
        return value


class TestResult(BaseModel):
    tests: list[TestCase] = Field(default_factory=list)


class RefactorChange(BaseModel):
    title: str
    description: str
    rationale: str
    addresses: list[str] = Field(
        default_factory=list,
        description="Titles of findings this change resolves.",
    )


class RefactorResult(BaseModel):
    summary: str
    refactored_code: str
    changes: list[RefactorChange] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    @field_validator("refactored_code")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("refactored_code must not be empty")
        return value


Verdict = Literal["approve", "needs_review", "reject"]


class FinalReview(BaseModel):
    approved: bool
    score: int = Field(ge=0, le=100)
    verdict: Verdict = "needs_review"
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    remaining_issues: list[str] = Field(default_factory=list)
    regression_risk: Literal["low", "medium", "high"] = "medium"
    recommendation: str
