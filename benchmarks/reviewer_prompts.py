"""Baseline reviewer prompt.

The project's headline claim is that an adversarial reviewer, told to default
to "not approved" and given both versions side by side, catches behaviour
changes that a plain comparison misses.

That is a testable claim, so it gets a baseline. This is the reviewer prompt
you would write without thinking about it: compare the two, tell me whether the
refactor is safe.

It is intentionally not strawmanned. It asks for the same JSON contract and
mentions behaviour preservation, because a baseline that ignores the task
entirely would make any prompt look good.
"""

from __future__ import annotations

NAIVE_REVIEWER = """Compare the original code with the refactored version.

ORIGINAL:
{original}

REFACTORED:
{refactored}

Tell me whether the refactor preserves behaviour. Return JSON with `approved`
(boolean), `summary` (string), and `recommendation` (string).
"""


def naive_reviewer_prompt(original: str, refactored: str) -> str:
    """Build the baseline comparison prompt."""
    return NAIVE_REVIEWER.format(original=original, refactored=refactored)
