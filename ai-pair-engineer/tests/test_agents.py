"""Test agent behavior with mocked LLM responses."""

import sys
from pathlib import Path

# Ensure the project is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch

import pytest

from ai_pair_engineer.models.schemas import (
    Finding,
    AnalysisResult,
    TestCase,
    TestResult,
    RefactorChange,
    RefactorResult,
    FinalReview,
)

from ai_pair_engineer.agents.analyzer import analyze_code
from ai_pair_engineer.agents.tester import generate_tests
from ai_pair_engineer.agents.refactor import refactor_code
from ai_pair_engineer.agents.reviewer import review_refactor


# Sample code for testing
SAMPLE_PYTHON = """
def process_users(users):
    results = []

    for user in users:
        if user["age"] >= 18:
            if user["email"] != "":
                name = user["name"].strip().lower()
                email = user["email"].strip().lower()

                if "@" in email:
                    results.append({
                        "name": name,
                        "email": email,
                        "adult": True
                    })

    return results
"""


def test_analyzer_with_mocked_response():
    """Test analyzer returns valid AnalysisResult with mocked LLM."""
    mock_response = (
        '{"summary": "Analysis complete", "findings": ['
        '{"severity": "high", "category": "code_smell", "title": "Long function",'
        '"description": "Function is too long", "recommendation": "Extract methods",'
        '"confidence": 0.9}]}'
    )

    with patch(
        "ai_pair_engineer.agents.analyzer.ask_llm", return_value=mock_response
    ):
        result = analyze_code("python", SAMPLE_PYTHON)
        assert isinstance(result, AnalysisResult)
        assert len(result.findings) == 1
        assert result.findings[0].title == "Long function"


def test_tester_with_mocked_response():
    """Test tester returns valid TestResult with mocked LLM."""
    mock_response = (
        '{"tests": [{"name": "test_valid", "purpose": "Test valid input",'
        '"test_code": "assert True"}]}'
    )

    with patch(
        "ai_pair_engineer.agents.tester.ask_llm", return_value=mock_response
    ):
        from ai_pair_engineer.models.schemas import AnalysisResult
        analysis = AnalysisResult(summary="", findings=[])
        result = generate_tests("python", SAMPLE_PYTHON, analysis.findings)
        assert isinstance(result, TestResult)
        assert len(result.tests) == 1
        assert result.tests[0].name == "test_valid"


def test_refactor_with_mocked_response():
    """Test refactor returns valid RefactorResult with mocked LLM."""
    mock_response = (
        '{"refactored_code": "def process_users(users):\\n    pass",'
        '"changes": [{"title": "Extract", "description": "Simplified",'
        '"rationale": "Test"}], "risks": []}'
    )

    with patch(
        "ai_pair_engineer.agents.refactor.ask_llm", return_value=mock_response
    ):
        from ai_pair_engineer.models.schemas import AnalysisResult
        analysis = AnalysisResult(summary="", findings=[])
        result = refactor_code("python", SAMPLE_PYTHON, analysis.findings)
        assert isinstance(result, RefactorResult)
        assert "def process_users" in result.refactored_code


def test_reviewer_with_mocked_response():
    """Test reviewer returns valid FinalReview with mocked LLM."""
    mock_response = (
        '{"approved": true, "score": 85,'
        '"strengths": ["Good structure"], "remaining_issues": [],'
        '"recommendation": "approve"}'
    )

    with patch(
        "ai_pair_engineer.agents.reviewer.ask_llm", return_value=mock_response
    ):
        from ai_pair_engineer.models.schemas import (
            AnalysisResult,
            RefactorResult,
            TestResult,
        )
        orig = "def foo(): pass"
        refactored = "def foo():\n    pass"
        tests = [TestCase(name="t1", purpose="p1", test_code="code1")]
        analysis = AnalysisResult(summary="", findings=[])
        refactor = RefactorResult(
            summary="",
            refactored_code=refactored,
            changes=[],
            risks=[],
        )
        result = review_refactor(orig, refactored, tests, analysis.findings)
        assert isinstance(result, FinalReview)
        assert result.approved is True
        assert result.score == 85


def test_analyzer_empty_code():
    """Test analyzer raises error on empty code."""
    with pytest.raises(ValueError, match="Source code cannot be empty"):
        analyze_code("python", "")


def test_tester_empty_code():
    """Test tester raises error on empty code."""
    from ai_pair_engineer.models.schemas import AnalysisResult
    with pytest.raises(ValueError, match="Source code cannot be empty"):
        generate_tests("python", "", AnalysisResult(summary="", findings=[]))


def test_refactor_empty_code():
    """Test refactor raises error on empty code."""
    from ai_pair_engineer.models.schemas import AnalysisResult
    with pytest.raises(ValueError, match="Source code cannot be empty"):
        refactor_code("python", "", AnalysisResult(summary="", findings=[]))


def test_reviewer_empty_original():
    """Test reviewer raises error on empty original code."""
    from ai_pair_engineer.models.schemas import (
        AnalysisResult,
        RefactorResult,
        TestResult,
    )
    with pytest.raises(ValueError, match="Original code cannot be empty"):
        review_refactor(
            "",
            "def foo(): pass",
            [TestCase(name="t1", purpose="p1", test_code="code1")],
            [],
        )