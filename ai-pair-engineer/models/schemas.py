from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Category(str, Enum):
    CODE_SMELL = "code_smell"
    MAINTAINABILITY = "maintainability"
    READABILITY = "readability"
    ARCHITECTURE = "architecture"
    COMPLEXITY = "complexity"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DUPLICATION = "duplication"


class Finding(BaseModel):
    severity: Severity
    category: Category
    title: str
    description: str
    recommendation: str
    confidence: float = 1.0


class AnalysisResult(BaseModel):
    summary: str
    findings: List[Finding]


class TestCase(BaseModel):
    name: str
    purpose: str
    test_code: str


class TestResult(BaseModel):
    tests: List[TestCase]


class RefactorChange(BaseModel):
    title: str
    description: str
    rationale: str


class RefactorResult(BaseModel):
    summary: str
    refactored_code: str
    changes: List[RefactorChange]
    risks: List[str]


class FinalReview(BaseModel):
    approved: bool
    score: int
    strengths: List[str]
    remaining_issues: List[str]
    recommendation: str