from __future__ import annotations

from typing import Any, list

from pydantic import ValidationError

from ...models.schemas import FinalReview, RefactorResult, AnalysisResult
from ...services.llm import ask_llm, parse_structured_response

logger = logging.getLogger(__name__)


def review_refactor(
    original_code: str,
    refactored_code: str,
    generated_tests: list,
    analyzer_findings: list,
) -> FinalReview:
    """Review a refactor against the original code.

    The Reviewer receives:
    - original source
    - refactored source
    - generated tests
    - important findings from the analyzer

    Each piece of context has a specific purpose and is not duplicated
    across all stages.
    """
    if not original_code or not original_code.strip():
        raise ValueError("Original code cannot be empty")
    if not refactored_code or not refactored_code.strip():
        raise ValueError("Refactored code cannot be empty")

    # Build context deliberately for the reviewer
    context_parts: list[str] = []
    context_parts.append("ORIGINAL CODE:")
    context_parts.append(original_code)
    context_parts.append("\nREFACTORED CODE:")
    context_parts.append(refactored_code)

    if generated_tests:
        context_parts.append("\nGenerated tests:")
        # Only include test names and purposes, not full code
        for test in generated_tests[:5]:
            context_parts.append(f"  - {test.name}: {test.purpose}")

    if analyzer_findings:
        context_parts.append("\nKey analyzer findings:")
        # Only pass findings relevant to the review
        relevant = [
            f for f in analyzer_findings
            if f.category in {"security", "critical", "high"}
        ][:5]
        for f in relevant:
            context_parts.append(
                f"  - {f.category.value}: {f.title} (severity: {f.severity})"
            )

    context_parts.append(
        "Compare original and refactored implementations. Check:"
    )
    context_parts.append("- behavior preservation")
    context_parts.append("- correctness")
    context_parts.append("- maintainability")
    context_parts.append("- complexity")
    context_parts.append("- testability")
    context_parts.append("- security")
    context_parts.append("- performance")
    context_parts.append("- regressions")
    context_parts.append(
        "Do not approve a refactor merely because it is stylistically different."
    )

    context = "\n".join(context_parts)

    system_prompt = open("ai-pair-engineer/prompts/reviewer.txt").read()
    user_prompt = context

    raw_response = ask_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,  # lowest temperature for deterministic review
    )

    try:
        parsed = parse_structured_response(raw_response, FinalReview)
        return parsed
    except (ValueError, ValidationError) as e:
        logger.warning("Failed to parse reviewer response: %s", e)
        system_prompt += (
            "\n\nThe previous response could not be parsed. "
            "Return only valid JSON matching the expected schema."
        )
        raw_response = ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        parsed = parse_structured_response(raw_response, FinalReview)
        return parsed