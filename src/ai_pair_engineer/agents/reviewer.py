"""Stage 4: adversarial review of the refactor against the original."""

from __future__ import annotations

from ai_pair_engineer.agents.base import (
    MAX_FINDINGS_FOR_REVIEW,
    MAX_TESTS_FOR_REVIEW,
    Agent,
    format_findings,
    require_source,
)
from ai_pair_engineer.models.schemas import FinalReview, Finding, Severity, TestCase

# The original filtered findings by ``category in {"security", "critical",
# "high"}`` - one category name and two severity names mixed into a single
# comparison against ``category``. It matched only security findings and
# silently dropped every critical or high finding of any other category. This
# stage now filters on severity directly.


class ReviewerAgent(Agent[FinalReview]):
    """Compares original against refactored and looks for regressions.

    This stage is deliberately hostile. It runs at temperature 0 and is asked to
    prove that behaviour changed, not to confirm that the refactor is nice.
    """

    prompt_name = "reviewer"
    schema = FinalReview
    stage_name = "Final Reviewer"

    def __init__(
        self,
        original_code: str,
        refactored_code: str,
        generated_tests: list[TestCase] | None = None,
        analyzer_findings: list[Finding] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(model=model)
        self.original_code = require_source(original_code, field_name="Original code")
        self.refactored_code = require_source(refactored_code, field_name="Refactored code")
        self.generated_tests = generated_tests or []
        self.analyzer_findings = analyzer_findings or []

    @property
    def temperature(self) -> float:
        return 0.0

    def _tests_in_scope(self) -> list[TestCase]:
        """Tests within the context budget.

        Capped so a large generated suite cannot crowd out the two code
        versions, which are what the reviewer actually reasons about.
        """
        return self.generated_tests[:MAX_TESTS_FOR_REVIEW]

    def build_user_prompt(self) -> str:
        sections = [
            "ORIGINAL CODE:",
            self.original_code,
            "",
            "REFACTORED CODE:",
            self.refactored_code,
        ]

        if self.generated_tests:
            sections += [
                "",
                "Tests written against the original code. Check that the refactored "
                "code still satisfies each one:",
                *(f"  - {test.name}: {test.purpose}" for test in self._tests_in_scope()),
            ]

        blocking = [
            finding
            for finding in self.analyzer_findings
            if finding.severity in {Severity.CRITICAL, Severity.HIGH}
        ][:MAX_FINDINGS_FOR_REVIEW]
        if blocking:
            sections += [
                "",
                "Unresolved analyzer findings that must be fixed, not refactored around:",
                *format_findings(blocking, with_description=False),
            ]

        sections += [
            "",
            "Check, in order of priority: behaviour preservation, correctness, "
            "regressions, error handling, then maintainability.",
            "Approve only if you can show the refactor is behaviour-preserving. "
            "A stylistic difference is not a reason to reject, and a stylistic "
            "difference is not a reason to approve.",
        ]
        return "\n".join(sections)


def review_refactor(
    original_code: str,
    refactored_code: str,
    generated_tests: list[TestCase] | None = None,
    analyzer_findings: list[Finding] | None = None,
) -> FinalReview:
    """Review a refactor against the original implementation.

    Raises:
        ValueError: if either source is empty.
        StageError: if the model cannot produce a valid FinalReview.
    """
    return ReviewerAgent(
        original_code,
        refactored_code,
        generated_tests,
        analyzer_findings,
    ).run()
