"""Stage 2: test generation, grounded in analyzer findings."""

from __future__ import annotations

from ai_pair_engineer.agents.base import (
    MAX_FINDINGS_FOR_TESTS,
    Agent,
    findings_for,
    format_findings,
    require_source,
)
from ai_pair_engineer.models.schemas import TEST_CATEGORIES, Finding, TestResult
from ai_pair_engineer.static import analyze_source


class TesterAgent(Agent[TestResult]):
    """Writes tests that target the failure modes the analyzer actually found.

    Error-handling and security findings are in scope because those are the
    paths a hand-written test suite usually skips. Style findings are not
    forwarded: no test can meaningfully cover a naming preference.
    """

    prompt_name = "tester"
    schema = TestResult
    stage_name = "Test Engineer"

    def __init__(
        self,
        language: str,
        source_code: str,
        analyzer_findings: list[Finding] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        self.language = language
        self.source_code = require_source(source_code)
        self.analyzer_findings = analyzer_findings or []
        self.static_report = analyze_source(self.source_code, self.language)

    def build_user_prompt(self) -> str:
        sections = [
            f"Language: {self.language}",
            "",
            "Source code:",
            self.source_code,
        ]

        relevant = findings_for(
            self.analyzer_findings,
            TEST_CATEGORIES,
            limit=MAX_FINDINGS_FOR_TESTS,
        )
        if relevant:
            sections += [
                "",
                "Known defects that these tests must cover:",
                *format_findings(relevant, with_description=True),
            ]

        if self.static_report.untested_function_names:
            sections += [
                "",
                "Functions requiring coverage: "
                + ", ".join(self.static_report.untested_function_names),
            ]

        sections += [
            "",
            "Cover normal behaviour, boundary values, invalid input, and failure "
            "paths. Use the standard test framework for the language.",
            "Do not test behaviour that cannot be inferred from the code.",
        ]
        return "\n".join(sections)


def generate_tests(
    language: str,
    source_code: str,
    analyzer_findings: list[Finding] | None = None,
) -> TestResult:
    """Generate test cases for the given code.

    Raises:
        ValueError: if ``source_code`` is empty.
        StageError: if the model cannot produce a valid TestResult.
    """
    return TesterAgent(language, source_code, analyzer_findings).run()
