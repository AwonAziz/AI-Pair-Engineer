"""The four pipeline stages."""

from ai_pair_engineer.agents.analyzer import AnalyzerAgent, analyze_code
from ai_pair_engineer.agents.base import (
    PipelineTrace,
    StageError,
    StageUsage,
    findings_for,
    require_source,
)
from ai_pair_engineer.agents.refactor import RefactorAgent, refactor_code
from ai_pair_engineer.agents.reviewer import ReviewerAgent, review_refactor
from ai_pair_engineer.agents.tester import TesterAgent, generate_tests

__all__ = [
    "AnalyzerAgent",
    "PipelineTrace",
    "RefactorAgent",
    "ReviewerAgent",
    "StageError",
    "StageUsage",
    "TesterAgent",
    "analyze_code",
    "findings_for",
    "generate_tests",
    "refactor_code",
    "require_source",
    "review_refactor",
]
