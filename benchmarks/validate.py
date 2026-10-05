"""Corpus integrity checks.

A benchmark can only be as trustworthy as its ground truth, and ground truth
rots silently. A defect annotated at line 14 after three lines were inserted
above it now points at the wrong code, and the benchmark keeps reporting a
number as though nothing happened.

These checks run in CI with no API key. They are cheap and they catch the
failure mode that matters most: an annotation that has drifted from the code it
describes.
"""

from __future__ import annotations

from benchmarks.corpus import load_cases, load_regressions, validate_case

# Expected phrases that are too generic to be evidence of anything. A reviewer
# saying "looks good" or "consider adding tests" matches these by accident, so
# a case relying on them would score a miss as a catch.
GENERIC_PHRASES = frozenset(
    {
        "fine",
        "ok",
        "okay",
        "good",
        "better",
        "worse",
        "change",
        "changes",
        "code",
        "bug",
        "issue",
        "problem",
        "improve",
        "improvement",
        "refactor",
        "test",
        "tests",
        "style",
        "readable",
        "clean",
        "safe",
        "works",
        "correct",
        "logic",
        "maintainable",
    }
)

# A defect matched on a single keyword group will be credited by a finding that
# merely names the same subsystem. Two or more groups is what makes a match
# specific enough to count.
MIN_KEYWORD_GROUPS = 2


def validate_corpus() -> tuple[list[str], int, int]:
    """Validate every case and regression.

    Returns:
        ``(problems, case_count, defect_count)``. An empty ``problems`` list
        means the corpus is internally consistent.
    """
    problems: list[str] = []
    cases = load_cases()
    defect_count = 0

    for case in cases:
        problems.extend(validate_case(case))
        defect_count += len(case.defects)

        if case.clean and not case.source.strip():
            problems.append(f"clean case {case.id!r} has an empty source")

        for defect in case.defects:
            if defect.keywords and len(defect.keywords) < MIN_KEYWORD_GROUPS:
                problems.append(
                    f"case {case.id!r} defect {defect.id!r} has "
                    f"{len(defect.keywords)} keyword group(s); at least "
                    f"{MIN_KEYWORD_GROUPS} are needed for a match to mean anything"
                )

            for group in defect.keywords:
                if any(len(word.strip()) < 2 for word in group):
                    problems.append(
                        f"case {case.id!r} defect {defect.id!r} has a keyword "
                        "shorter than two characters, which will match anything"
                    )

    seen_ids: set[str] = set()
    for case in cases:
        if case.id in seen_ids:
            problems.append(f"duplicate case id {case.id!r}")
        seen_ids.add(case.id)

        defect_ids = [defect.id for defect in case.defects]
        if len(defect_ids) != len(set(defect_ids)):
            problems.append(f"case {case.id!r} has duplicate defect ids")

        for defect in case.defects:
            if case.defect_by_id(defect.id) is None:
                problems.append(f"case {case.id!r} defect {defect.id!r} is unresolvable")

    problems.extend(_validate_regressions())

    return problems, len(cases), defect_count


def _validate_regressions() -> list[str]:
    """Check that each regression pair differs only as intended."""
    problems: list[str] = []
    cases = load_regressions()
    seen: set[str] = set()

    for case in cases:
        if case.id in seen:
            problems.append(f"duplicate regression id {case.id!r}")
        seen.add(case.id)

        if case.original == case.refactored:
            problems.append(f"regression {case.id!r} has identical original and refactored code")
            continue

        original_lines = _normalized(case.original)
        refactored_lines = _normalized(case.refactored)

        if original_lines == refactored_lines:
            problems.append(
                f"regression {case.id!r} differs only in whitespace or comments; "
                "that is a refactor with no behaviour change to catch"
            )

        if not case.change:
            problems.append(f"regression {case.id!r} has no `change` description")

        if not case.expected:
            problems.append(f"regression {case.id!r} has no expected phrases")

        # An expected phrase is matched against the *reviewer's output*, so the
        # risk is not that it appears in the code but that it is generic enough
        # to appear in any review at all. "looks fine" would credit a reviewer
        # that noticed nothing.
        for phrase in case.expected:
            if phrase.strip().lower() in GENERIC_PHRASES:
                problems.append(
                    f"regression {case.id!r} expects the phrase {phrase!r}, which "
                    "is too generic to be evidence that the change was noticed"
                )

    return problems


def _normalized(source: str) -> list[str]:
    """Strip comments and blank lines so only real code differences remain."""
    lines: list[str] = []
    for raw in source.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines
