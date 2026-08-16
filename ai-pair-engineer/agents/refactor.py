from __future__ import annotations

from typing import Any, list

from pydantic import ValidationError

from ...models.schemas import RefactorResult, RefactorChange
from ...services.llm import ask_llm, parse_structured_response

logger = logging.getLogger(__name__)


def refactor_code(language: str, source_code: str, analyzer_findings: list) -> RefactorResult:
    """Refactor the code based on analyzer findings.

    The Refactor receives:
    - language
    - source code
    - analyzer findings (only relevant findings about maintainability, readability, etc.)

    Context budgeting: only findings related to maintainability, readability,
    and structure are passed. Security and performance findings are summarized
    but not dwelt upon unless they relate to structure.
    """
    if not source_code or not source_code.strip():
        raise ValueError("Source code cannot be empty")

    # Filter to findings relevant to refactoring maintainability/readability
    relevant_findings = [
        f for f in analyzer_findings
        if f.category in {
            "code_smell", "maintainability", "readability",
            "architecture", "complexity", "duplication"
        }
    ][:10]  # limit to top 10 most relevant

    # Build context: only pass what's relevant to refactoring
    context_parts: list[str] = []
    context_parts.append(f"Language: {language}")
    context_parts.append("Source code:")
    context_parts.append(source_code)

    if relevant_findings:
        context_parts.append("\nAnalyzer findings to address via refactoring:")
        for finding in relevant_findings:
            context_parts.append(
                f"  - {finding.category.value}: {finding.title} - "
                f"{finding.description}"
            )
        context_parts.append(
            "Improve maintainability and readability while preserving intended behavior."
        )
        context_parts.append("Do not introduce unnecessary abstractions.")

    context = "\n".join(context_parts)

    system_prompt = open("ai-pair-engineer/prompts/refactor.txt").read()
    user_prompt = context

    raw_response = ask_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
    )

    try:
        parsed = parse_structured_response(raw_response, RefactorResult)
        return parsed
    except (ValueError, ValidationError) as e:
        logger.warning("Failed to parse refactor response: %s", e)
        system_prompt += (
            "\n\nThe previous response could not be parsed. "
            "Return only valid JSON matching the expected schema."
        )
        raw_response = ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        parsed = parse_structured_response(raw_response, RefactorResult)
        return parsed