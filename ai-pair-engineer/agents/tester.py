from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ...models.schemas import TestResult, TestCase
from ...services.llm import ask_llm, parse_structured_response

logger = logging.getLogger(__name__)


def generate_tests(language: str, source_code: str, analyzer_findings: list) -> TestResult:
    """Generate test cases for the given code.

    The Tester receives:
    - language
    - source code
    - analyzer findings (relevant issues to cover in tests)

    Only task-relevant context is passed. Unrelated findings are excluded.
    """
    if not source_code or not source_code.strip():
        raise ValueError("Source code cannot be empty")

    # Filter to relevant findings for test generation
    relevant_categories = {
        "error_handling",
        "security",
        "boundary_conditions",
        "invalid_inputs",
    }
    relevant_findings = [
        f for f in analyzer_findings
        if f.category.value in relevant_categories
    ]

    # Build context: only pass what's relevant
    context_parts: list[str] = []
    context_parts.append(f"Language: {language}")
    context_parts.append("Source code:")
    context_parts.append(source_code)

    if relevant_findings:
        context_parts.append("\nAnalyzer findings relevant to testing:")
        for finding in relevant_findings[:5]:  # limit to top 5
            context_parts.append(
                f"  - {finding.category.value}: {finding.title} "
                f"(severity: {finding.severity})"
            )

    context_parts.append("\nGenerate tests covering: normal behavior, edge cases, "
                         "invalid inputs, failure paths, boundary conditions.")
    context_parts.append("Do not generate tests for behavior that cannot be "
                         "inferred from the code.")

    context = "\n".join(context_parts)

    system_prompt = open("ai-pair-engineer/prompts/tester.txt").read()
    user_prompt = context

    raw_response = ask_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
    )

    try:
        parsed = parse_structured_response(raw_response, TestResult)
        return parsed
    except (ValueError, ValidationError) as e:
        logger.warning("Failed to parse tester response: %s", e)
        system_prompt += (
            "\n\nThe previous response could not be parsed. "
            "Return only valid JSON matching the expected schema."
        )
        raw_response = ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        parsed = parse_structured_response(raw_response, TestResult)
        return parsed