"""Shared machinery for the four pipeline stages.

Each stage is the same shape: build a scoped context, call the model, validate
against a schema, and retry once with a correction prompt if validation fails.
That shape lives here so the stages only describe what makes them different.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ai_pair_engineer.models.schemas import Category, Finding, Severity
from ai_pair_engineer.prompts import load_prompt
from ai_pair_engineer.services.llm import (
    LLMResponseError,
    TokenUsage,
    ask_llm,
    estimate_cost_usd,
    parse_structured_response,
)

logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT", bound=BaseModel)

# Analysis stages stay at 0.1 for consistency; only the reviewer drops to 0.0.
DEFAULT_TEMPERATURE = 0.1

# Context budgeting caps. These are the mechanism behind the "scoped context"
# claim: each stage sees only what it needs, capped so a pathological input
# cannot blow the token budget.
MAX_FINDINGS_FOR_TESTS = 6
MAX_FINDINGS_FOR_REFACTOR = 8
MAX_TESTS_FOR_REVIEW = 8
MAX_FINDINGS_FOR_REVIEW = 6


class StageError(RuntimeError):
    """A stage could not produce a valid result."""


@dataclass
class StageUsage:
    """Token and call accounting for one stage run."""

    stage: str
    model: str
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def record(self, usage: TokenUsage) -> None:
        """Accumulate one completed provider call."""
        self.attempts += 1
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        self.cost_usd += estimate_cost_usd(usage)
        if usage.model:
            self.model = usage.model

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "attempts": self.attempts,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class PipelineTrace:
    """Per-stage accounting for a whole pipeline run."""

    stages: list[StageUsage] = field(default_factory=list)

    def record(self, usage: StageUsage) -> None:
        self.stages.append(usage)

    @property
    def total_tokens(self) -> int:
        return sum(stage.total_tokens for stage in self.stages)

    @property
    def total_calls(self) -> int:
        return sum(stage.attempts for stage in self.stages)

    @property
    def total_cost_usd(self) -> float:
        return sum(stage.cost_usd for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


def require_source(source_code: str, *, field_name: str = "Source code") -> str:
    """Validate that submitted code is present before spending a model call.

    Raises:
        ValueError: if the code is empty or whitespace only.
    """
    if not source_code or not source_code.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return source_code


def findings_for(
    findings: list[Finding],
    categories: frozenset[Category],
    *,
    limit: int,
) -> list[Finding]:
    """Select findings relevant to a stage, most severe first.

    Severity ordering matters: the caps above would otherwise truncate the
    findings that matter most, since models tend to list low-severity notes
    last but not reliably.
    """
    selected = [finding for finding in findings if finding.category in categories]
    selected.sort(key=lambda f: _severity_rank(f.severity), reverse=True)
    return selected[:limit]


def _severity_rank(severity: Severity) -> int:
    order = {
        Severity.CRITICAL: 4,
        Severity.HIGH: 3,
        Severity.MEDIUM: 2,
        Severity.LOW: 1,
    }
    return order[severity]


def format_findings(findings: list[Finding], *, with_description: bool) -> list[str]:
    """Render findings as prompt lines.

    ``with_description`` is False for the test stage: a test author needs to know
    what failed, not a paragraph of restated context.
    """
    lines: list[str] = []
    for finding in findings:
        location = finding.location.render() if finding.location else ""
        suffix = f" [{location}]" if location else ""
        if with_description:
            lines.append(
                f"  - [{finding.severity.value}/{finding.category.value}] "
                f"{finding.title}{suffix}: {finding.description}"
            )
        else:
            lines.append(
                f"  - [{finding.severity.value}] {finding.title}{suffix} "
                f"(category: {finding.category.value})"
            )
    return lines


class Agent(ABC, Generic[ResultT]):
    """Base class for a single-call, schema-validated pipeline stage."""

    #: Prompt file stem, without extension.
    prompt_name: str
    #: Pydantic model the stage must produce.
    schema: type[ResultT]
    #: Human-readable name used in logs and the trace.
    stage_name: str

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        usage: StageUsage | None = None,
    ) -> None:
        self.model = model
        self._temperature_override = temperature
        # Every stage carries its own accounting so a caller can read the cost
        # of one stage without reconstructing the whole trace.
        self.usage = usage or StageUsage(stage=self.stage_name, model=model or "(default)")

    @abstractmethod
    def build_user_prompt(self) -> str:
        """Assemble the scoped context this stage is allowed to see."""

    @property
    def system_prompt(self) -> str:
        return load_prompt(self.prompt_name)

    @property
    def temperature(self) -> float:
        """Sampling temperature for this stage.

        A subclass override (``DEFAULT_TEMPERATURE``) is used unless the caller
        passes one explicitly. Benchmarks pass 0 to make runs comparable.
        """
        if self._temperature_override is not None:
            return self._temperature_override
        return self.default_temperature

    @property
    def default_temperature(self) -> float:
        return DEFAULT_TEMPERATURE

    def run(self) -> ResultT:
        """Execute the stage and return a validated result.

        Raises:
            StageError: if both the initial attempt and the correction retry
                fail to satisfy the schema.
        """
        user_prompt = self.build_user_prompt()
        correction = ""

        for attempt in (1, 2):
            raw = ask_llm(
                system_prompt=self.system_prompt + correction,
                user_prompt=user_prompt,
                model=self.model,
                temperature=self.temperature,
                on_usage=self.usage.record,
            )
            try:
                return parse_structured_response(raw, self.schema)
            except LLMResponseError as exc:
                logger.warning(
                    "%s stage failed validation on attempt %s: %s",
                    self.stage_name,
                    attempt,
                    exc,
                )
                if attempt == 2:
                    raise StageError(
                        f"{self.stage_name} stage produced an invalid result after "
                        f"a correction retry: {exc}"
                    ) from exc
                correction = CORRECTION_SUFFIX

        raise StageError(f"{self.stage_name} stage exhausted its retries")  # pragma: no cover


CORRECTION_SUFFIX = """

Your previous response did not match the required schema and was rejected.

Reply with ONLY a JSON object. No prose before or after, no markdown fences.
Every required field must be present and use exactly the documented value set.
Do not add fields that are not in the schema. Do not omit fields that are.
"""
