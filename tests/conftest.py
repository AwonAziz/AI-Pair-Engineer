"""Shared fixtures.

Every test runs against mocked model responses, so the suite needs no API key
and makes no network calls. The fake client is patched at the module boundary
each agent imports from, which is the only seam that matters here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_pair_engineer.models.schemas import (
    AnalysisResult,
    FinalReview,
    RefactorResult,
    TestResult,
)
from ai_pair_engineer.services import llm as llm_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT_ROOT / "examples"

SAMPLE_SOURCE = """def process_users(users):
    results = []
    for user in users:
        if user["age"] >= 18:
            if user["email"] != "":
                email = user["email"].strip().lower()
                if "@" in email:
                    results.append({"email": email, "adult": True})
    return results
"""

ANALYSIS_JSON = json.dumps(
    {
        "summary": "Two correctness defects and one maintainability problem.",
        "findings": [
            {
                "severity": "high",
                "category": "error_handling",
                "title": "Unvalidated dict access raises KeyError",
                "description": "user['age'] is read without checking the key exists.",
                "recommendation": "Use user.get('age') and skip malformed records.",
                "confidence": 0.95,
                "location": {"line": 3, "end_line": 3, "symbol": "process_users"},
            },
            {
                "severity": "medium",
                "category": "complexity",
                "title": "Nested conditionals flatten poorly",
                "description": "Four levels of nesting for one filter.",
                "recommendation": "Use guard clauses.",
                "confidence": 0.8,
                "location": {"line": 3, "symbol": "process_users"},
            },
            {
                "severity": "low",
                "category": "readability",
                "title": "Inconsistent key naming",
                "description": "Results mix email and adult without a typed shape.",
                "recommendation": "Introduce a dataclass.",
                "confidence": 0.6,
            },
        ],
    }
)

TESTS_JSON = json.dumps(
    {
        "tests": [
            {
                "name": "test_missing_age_key_is_skipped",
                "purpose": "A record without an age must not raise KeyError.",
                "test_code": "def test_missing_age():\n    assert process_users([{}]) == []\n",
                "covers": ["Unvalidated dict access raises KeyError"],
            },
            {
                "name": "test_blank_email_excluded",
                "purpose": "Records with an empty email are filtered out.",
                "test_code": (
                    "def test_blank():\n"
                    "    assert process_users([{'age': 20, 'email': ''}]) == []\n"
                ),
            },
        ]
    }
)

REFACTOR_JSON = json.dumps(
    {
        "summary": "Replaced nested conditionals with guard clauses.",
        "refactored_code": (
            "def process_users(users):\n"
            "    results = []\n"
            "    for user in users:\n"
            "        if not user.get('age'):\n"
            "            continue\n"
            "        results.append(user)\n"
            "    return results\n"
        ),
        "changes": [
            {
                "title": "Guard clause for age check",
                "description": "Inverted the age condition and continued early.",
                "rationale": "Removes two levels of nesting.",
                "addresses": ["Nested conditionals flatten poorly"],
            }
        ],
        "risks": ["user.get returns None for a missing key, which differs from KeyError."],
    }
)

REVIEW_JSON = json.dumps(
    {
        "approved": False,
        "verdict": "needs_review",
        "score": 72,
        "summary": "The guard clause changes behaviour for records missing an age key.",
        "strengths": ["Nesting reduced from four levels to one"],
        "remaining_issues": [
            "Original raised KeyError on a missing age key; the refactor silently skips it.",
        ],
        "regression_risk": "high",
        "recommendation": "Preserve the KeyError or update the callers that relied on it.",
    }
)

CANNED_RESPONSES = {
    "analysis": ANALYSIS_JSON,
    "tests": TESTS_JSON,
    "refactor": REFACTOR_JSON,
    "review": REVIEW_JSON,
}


@pytest.fixture
def sample_source() -> str:
    return SAMPLE_SOURCE


@pytest.fixture
def example_file() -> Path:
    return EXAMPLES / "sample.py"


@pytest.fixture
def canned() -> dict[str, str]:
    return dict(CANNED_RESPONSES)


@pytest.fixture(autouse=True)
def _no_real_api_calls(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly instead of quietly spending tokens.

    ``tests/test_llm.py`` exercises ``get_client`` directly and opts out. Every
    other test must patch ``ai_pair_engineer.agents.base.ask_llm``; if one
    reaches the network, this raises rather than making a real request.
    """
    if request.node.path.name == "test_llm.py":
        return

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            f"{request.node.name} attempted a real LLM call; patch "
            "ai_pair_engineer.agents.base.ask_llm instead"
        )

    monkeypatch.setattr(llm_module, "get_client", _explode)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def analysis_result() -> AnalysisResult:
    return AnalysisResult.model_validate_json(ANALYSIS_JSON)


@pytest.fixture
def test_result() -> TestResult:
    return TestResult.model_validate_json(TESTS_JSON)


@pytest.fixture
def refactor_result() -> RefactorResult:
    return RefactorResult.model_validate_json(REFACTOR_JSON)


@pytest.fixture
def review_result() -> FinalReview:
    return FinalReview.model_validate_json(REVIEW_JSON)
