"""Stage 1: static plus model analysis of the submitted code."""

from __future__ import annotations

from ai_pair_engineer.agents.base import Agent, require_source
from ai_pair_engineer.models.schemas import AnalysisResult
from ai_pair_engineer.static import StaticReport, analyze_source

MAX_SOURCE_CHARS = 20_000


class AnalyzerAgent(Agent[AnalysisResult]):
    """Identifies defects and maintainability problems in one source file.

    The stage sees the code plus the local static report. It does not see
    anything from downstream stages, and nothing downstream sees its raw
    prompt.
    """

    prompt_name = "analyzer"
    schema = AnalysisResult
    stage_name = "Code Analyzer"

    def __init__(
        self,
        language: str,
        source_code: str,
        static_report: StaticReport | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        include_static: bool = True,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        self.language = language
        self.source_code = require_source(source_code)
        self.include_static = include_static
        self.static_report = (
            static_report
            if static_report is not None
            else analyze_source(self.source_code, self.language)
        )

    def build_user_prompt(self) -> str:
        sections = [
            f"Language: {self.language}",
            "",
            "Source code:",
            _truncate(self.source_code),
        ]

        if self.include_static:
            sections += [
                "",
                "Deterministic static analysis (verified locally, trust the line numbers):",
                *self.static_report.summary_lines(),
            ]

            if self.static_report.untested_function_names:
                names = ", ".join(self.static_report.untested_function_names)
                sections += ["", f"Public functions needing coverage: {names}"]

        sections += [
            "",
            "Report only problems you can point at in this code. Include a line "
            "number and the enclosing function name wherever one applies.",
        ]
        return "\n".join(sections)


def _truncate(source_code: str) -> str:
    if len(source_code) <= MAX_SOURCE_CHARS:
        return source_code
    return (
        source_code[:MAX_SOURCE_CHARS] + f"\n\n... truncated at {MAX_SOURCE_CHARS} characters "
        f"({len(source_code)} submitted in total) ..."
    )


def analyze_code(language: str, source_code: str) -> AnalysisResult:
    """Analyze code and return structured findings.

    Kept as a module-level function for backwards compatibility with the
    original API and for use as a single-stage call.

    Raises:
        ValueError: if ``source_code`` is empty.
        StageError: if the model cannot produce a valid AnalysisResult.
    """
    return AnalyzerAgent(language, source_code).run()
