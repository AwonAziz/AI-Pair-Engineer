from __future__ import annotations

import ast
import json
import logging

from pydantic import ValidationError

from ...models.schemas import AnalysisResult, Finding
from ...services.llm import ask_llm, parse_structured_response

logger = logging.getLogger(__name__)


def _get_static_analysis(source_code: str, language: str) -> dict[str, Any]:
    """Perform lightweight static analysis using ast module for Python.

    Identifies simple structural information:
    - number of functions
    - number of classes
    - nesting depth where feasible
    - syntax errors
    - obvious complexity indicators
    """
    info: dict[str, Any] = {
        "num_functions": 0,
        "num_classes": 0,
        "max_nesting": 0,
        "syntax_error": None,
        "complexity_indicators": 0,
    }

    if language == "python":
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    info["num_functions"] += 1
                elif isinstance(node, ast.ClassDef):
                    info["num_classes"] += 1

            # Calculate max nesting depth
            max_depth = 0

            def _walk_depth(nodes, depth=0):
                nonlocal max_depth
                max_depth = max(max_depth, depth)
                for node in nodes:
                    if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.FunctionDef, ast.ClassDef)):
                        _walk_depth(ast.iter_child_nodes(node), depth + 1)

            _walk_depth(tree.body)
            info["max_nesting"] = max_depth

            # Count complexity indicators
            info["complexity_indicators"] = (
                len(list(ast.walk(tree)))  # total nodes as rough indicator
            )
        except SyntaxError as e:
            info["syntax_error"] = str(e)

    return info


def analyze_code(language: str, source_code: str) -> AnalysisResult:
    """Analyze code and return structured findings.

    The Analyzer receives only:
    - language
    - source code

    Static analysis is added as supplementary evidence, but the LLM
    remains responsible for semantic review.
    """
    if not source_code or not source_code.strip():
        raise ValueError("Source code cannot be empty")

    # Run static analysis for Python
    static_info = _get_static_analysis(source_code, language)

    # Build context for the LLM
    context_parts: list[str] = []

    context_parts.append(f"Language: {language}")
    context_parts.append("Source code:")
    context_parts.append(source_code)
    context_parts.append("\nStatic analysis evidence:")
    if static_info.get("syntax_error"):
        context_parts.append(f"  Syntax error: {static_info['syntax_error']}")
    context_parts.append(f"  Functions found: {static_info.get('num_functions', 0)}")
    context_parts.append(f"  Classes found: {static_info.get('num_classes', 0)}")
    context_parts.append(f"  Max nesting depth: {static_info.get('max_nesting', 0)}")
    context_parts.append(f"  Complexity indicators: {static_info.get('complexity_indicators', 0)}")

    context = "\n".join(context_parts)

    system_prompt = open("ai-pair-engineer/prompts/analyzer.txt").read()
    user_prompt = context

    raw_response = ask_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
    )

    try:
        parsed = parse_structured_response(raw_response, AnalysisResult)
        return parsed
    except (ValueError, ValidationError) as e:
        logger.warning("Failed to parse analyzer response: %s", e)
        # Retry with a correction prompt
        system_prompt += (
            "\n\nThe previous response could not be parsed. "
            "Return only valid JSON matching the expected schema."
        )
        raw_response = ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        parsed = parse_structured_response(raw_response, AnalysisResult)
        return parsed