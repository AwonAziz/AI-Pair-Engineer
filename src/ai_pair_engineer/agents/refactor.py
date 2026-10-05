"""Stage 3: behaviour-preserving refactoring."""

from __future__ import annotations

from ai_pair_engineer.agents.base import (
    MAX_FINDINGS_FOR_REFACTOR,
    Agent,
    findings_for,
    format_findings,
    require_source,
)

# The previous implementation compared `finding.category` (an enum member)
# against a set of bare strings. That works only because Severity and Category
# subclass `str`, so it is silently wrong the moment the enums stop doing so.
from ai_pair_engineer.models.schemas import REFACTOR_CATEGORIES, Finding, RefactorResult


class RefactorAgent(Agent[RefactorResult]):
    """Improves structure without changing behaviour.

    Receives only the maintainability-class findings. Passing security findings
    here would invite the model to "fix" a vulnerability mid-refactor, which is
    exactly the kind of change that hides a behavioural regression.
    """

    prompt_name = "refactor"
    schema = RefactorResult
    stage_name = "Refactoring Engineer"

    def __init__(
        self,
        language: str,
        source_code: str,
        analyzer_findings: list[Finding] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(model=model)
        self.language = language
        self.source_code = require_source(source_code)
        self.analyzer_findings = analyzer_findings or []

    def build_user_prompt(self) -> str:
        sections = [
            f"Language: {self.language}",
            "",
            "Source code:",
            self.source_code,
        ]

        relevant = findings_for(
            self.analyzer_findings,
            REFACTOR_CATEGORIES,
            limit=MAX_FINDINGS_FOR_REFACTOR,
        )
        if relevant:
            sections += [
                "",
                "Findings to address:",
                *format_findings(relevant, with_description=True),
            ]

        # The behaviour-preservation constraint is unconditional. It used to be
        # nested inside the `if relevant` block, so a clean file reached the
        # model with no instruction to preserve behaviour at all.
        sections += [
            "",
            "Preserve behaviour exactly. Do not introduce abstractions that the "
            "code does not need yet.",
        ]
        return "\n".join(sections)


def refactor_code(
    language: str,
    source_code: str,
    analyzer_findings: list[Finding] | None = None,
) -> RefactorResult:
    """Refactor code while preserving behaviour.

    Raises:
        ValueError: if ``source_code`` is empty.
        StageError: if the model cannot produce a valid RefactorResult.
    """
    return RefactorAgent(language, source_code, analyzer_findings).run()
