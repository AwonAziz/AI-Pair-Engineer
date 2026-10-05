"""Naive baseline prompts.

The point of a baseline is to answer "is the four-stage pipeline earning its
complexity?" Without one, every claim about prompt quality is unfalsifiable.

The naive prompt is deliberately the thing a person would write on their first
attempt: review this code and return JSON. One call, no system prompt telling
the model what to prioritise, no local evidence, no adversarial second pass.

Keeping it this simple is the honest choice. A baseline tuned to be weak would
make the pipeline look better than it is, and a reviewer who spots that would
discount the whole benchmark.
"""

from __future__ import annotations

NAIVE_ANALYZER = (
    "Review the following code and return a JSON object with a `summary` string "
    "and a `findings` array. Each finding needs `severity` (low, medium, high, "
    "or critical), `category`, `title`, `description`, and `recommendation`.\n\n"
    "{source}"
)


def naive_analyzer_prompt(source: str, language: str) -> str:
    """Build the single-shot baseline prompt for a source file."""
    return NAIVE_ANALYZER.format(source=f"Language: {language}\n\n{source}")
