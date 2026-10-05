"""Local static analysis that runs before any model call."""

from ai_pair_engineer.static.python_analyzer import (
    FunctionMetrics,
    StaticReport,
    analyze_source,
    module_mutables,
    to_evidence,
)

__all__ = [
    "FunctionMetrics",
    "StaticReport",
    "analyze_source",
    "module_mutables",
    "to_evidence",
]
