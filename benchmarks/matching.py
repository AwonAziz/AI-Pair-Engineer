"""Tie findings to planted defects.

Matching uses two independent rules, because each covers a different way a
reviewer can be right.

:attr:`MatchRule.LOCATED` fires when the finding names a line inside the
defect's span. A model that reports the right line has almost certainly found
the right thing.

:attr:`MatchRule.SEMANTIC` fires when every keyword group in the defect appears
somewhere in the finding's text. A model can find the bug and misjudge the line
number, or report no line at all, and still deserve credit.

Requiring a specific line is too brittle to be the only rule. Requiring
keywords alone is loose enough to credit coincidences, so a defect's keyword
groups are written to be specific.
"""

from __future__ import annotations

from ai_pair_engineer.models.schemas import Finding
from benchmarks.models import Match, MatchRule, PlantedDefect

# Text a finding is matched against. The recommendation is included because a
# reviewer often names the fix rather than the defect.
_SEARCH_FIELDS = ("title", "description", "recommendation")


def finding_text(finding: Finding) -> str:
    """Flatten a finding into the text used for keyword matching."""
    parts = [getattr(finding, field, "") for field in _SEARCH_FIELDS]
    if finding.location and finding.location.symbol:
        parts.append(finding.location.symbol)
    return " ".join(parts).lower()


def keyword_match(defect: PlantedDefect, text: str) -> bool:
    """Whether every keyword group in ``defect`` appears in ``text``.

    Both sides are lowercased here rather than trusting the caller. The runner
    happens to pass already-lowercased text, but this function is public, and a
    case-sensitive comparison that silently stops matching on a capitalised
    word would show up as a recall drop with no obvious cause.

    A defect with no keyword groups never matches on keywords alone, so it can
    only be credited by line. That is the safe direction: an unmatchable defect
    is recorded as a miss rather than silently credited.
    """
    if not defect.keywords:
        return False
    haystack = text.lower()
    return all(any(keyword.lower() in haystack for keyword in group) for group in defect.keywords)


def match_rule_for(
    finding: Finding,
    defect: PlantedDefect,
    text: str | None = None,
) -> MatchRule | None:
    """Return which rule ties ``finding`` to ``defect``, or None."""
    haystack = text if text is not None else finding_text(finding)
    located = (
        finding.location is not None
        and finding.location.line is not None
        and defect.contains_line(finding.location.line)
    )
    semantic = keyword_match(defect, haystack)

    if located and semantic:
        return MatchRule.BOTH
    if located:
        return MatchRule.LOCATED
    if semantic:
        return MatchRule.SEMANTIC
    return None


def match_findings(
    findings: list[Finding], defects: tuple[PlantedDefect, ...]
) -> tuple[list[Match], list[Finding]]:
    """Tie findings to defects.

    Returns the matches and the findings that matched nothing. A defect matched
    by several findings is credited once, and every matching finding is kept in
    the match list so repeated reports of the same defect stay visible rather
    than being collapsed away.
    """
    matches: list[Match] = []
    matched_findings: list[Finding] = []

    for finding in findings:
        text = finding_text(finding)
        for defect in defects:
            rule = match_rule_for(finding, defect, text)
            if rule is not None:
                matches.append(Match(defect_id=defect.id, finding_title=finding.title, rule=rule))
                matched_findings.append(finding)
                break

    unmatched = [finding for finding in findings if finding not in matched_findings]
    return matches, unmatched


def matched_defect_ids(matches: list[Match]) -> set[str]:
    """Distinct defect ids covered by ``matches``."""
    return {match.defect_id for match in matches}
