"""Benchmark harness.

Measures whether the analyzer finds defects that were deliberately planted,
and whether the adversarial reviewer catches behaviour changes a plain
comparison misses.

    python -m benchmarks run --tiers static,analyzer+static
    python -m benchmarks run --tiers all --regressions
    python -m benchmarks validate

Run ``validate`` to check the corpus annotations without spending anything.
It needs no API key, so it belongs in CI.
"""

from benchmarks.corpus import (
    CorpusError,
    load_cases,
    load_regressions,
    validate_case,
)
from benchmarks.matching import match_findings, match_rule_for
from benchmarks.models import (
    CaseResult,
    DetectionCase,
    Match,
    MatchRule,
    PlantedDefect,
    RegressionCase,
    RegressionResult,
    RunReport,
    Tier,
    TierScore,
)
from benchmarks.report import render_json, render_markdown
from benchmarks.runner import run_benchmark

__all__ = [
    "CaseResult",
    "CorpusError",
    "DetectionCase",
    "Match",
    "MatchRule",
    "PlantedDefect",
    "RegressionCase",
    "RegressionResult",
    "RunReport",
    "Tier",
    "TierScore",
    "load_cases",
    "load_regressions",
    "match_findings",
    "match_rule_for",
    "render_json",
    "render_markdown",
    "run_benchmark",
    "validate_case",
]
