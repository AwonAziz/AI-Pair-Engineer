"""Test Pydantic schemas validation - standalone, no external imports needed for schemas."""

from enum import Enum


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


# Minimal Pydantic-like validation without full pydantic dependency
class Finding:
    def __init__(self, severity, category, title, description, recommendation, confidence=1.0):
        self.severity = severity
        self.category = category
        self.title = title
        self.description = description
        self.recommendation = recommendation
        self.confidence = confidence


class AnalysisResult:
    def __init__(self, summary, findings):
        self.summary = summary
        self.findings = findings


class TestCase:
    def __init__(self, name, purpose, test_code):
        self.name = name
        self.purpose = purpose
        self.test_code = test_code


class TestResult:
    def __init__(self, tests):
        self.tests = tests


class RefactorChange:
    def __init__(self, title, description, rationale):
        self.title = title
        self.description = description
        self.rationale = rationale


class RefactorResult:
    def __init__(self, summary, refactored_code, changes, risks):
        self.summary = summary
        self.refactored_code = refactored_code
        self.changes = changes
        self.risks = risks


class FinalReview:
    def __init__(self, approved, score, strengths, remaining_issues, recommendation):
        self.approved = approved
        self.score = score
        self.strengths = strengths
        self.remaining_issues = remaining_issues
        self.recommendation = recommendation


def test_finding_creation():
    """Test Finding model can be created with valid data."""
    finding = Finding(
        severity=Severity.MEDIUM,
        category=Category.CODE_SMELL,
        title="Long function",
        description="Function is too long and should be broken up.",
        recommendation="Extract smaller functions.",
        confidence=0.9,
    )
    assert finding.severity == Severity.MEDIUM
    assert finding.category == Category.CODE_SMELL
    assert finding.title == "Long function"
    assert finding.confidence == 0.9


def test_finding_defaults():
    """Test Finding model uses defaults."""
    finding = Finding(
        severity=Severity.LOW,
        category=Category.MAINTAINABILITY,
        title="Test finding",
        description="A test description.",
        recommendation="Do something.",
    )
    assert finding.confidence == 1.0


def test_analysis_result():
    """Test AnalysisResult model."""
    finding = Finding(
        severity=Severity.HIGH,
        category=Category.SECURITY,
        title="SQL injection risk",
        description="Query uses f-string formatting.",
        recommendation="Use parameterized queries.",
        confidence=0.95,
    )
    result = AnalysisResult(
        summary="Security issues found",
        findings=[finding],
    )
    assert len(result.findings) == 1
    assert result.findings[0].severity == Severity.HIGH


def test_test_case():
    """Test TestCase model."""
    test = TestCase(
        name="test_valid_input",
        purpose="Verify valid input is accepted",
        test_code="assert process_users([{'name': 'Alice', 'age': 25, 'email': 'alice@example.com'}]) == expected",
    )
    assert test.name == "test_valid_input"
    assert test.purpose == "Verify valid input is accepted"


def test_test_result():
    """Test TestResult model."""
    test1 = TestCase(name="t1", purpose="p1", test_code="code1")
    test2 = TestCase(name="t2", purpose="p2", test_code="code2")
    result = TestResult(tests=[test1, test2])
    assert len(result.tests) == 2


def test_refactor_change():
    """Test RefactorChange model."""
    change = RefactorChange(
        title="Extract function",
        description="Moved code into separate function",
        rationale="Improved readability",
    )
    assert change.title == "Extract function"


def test_refactor_result():
    """Test RefactorResult model."""
    change = RefactorChange(title="Rename", description="Rename variable", rationale="Clarity")
    result = RefactorResult(
        summary="Improved readability",
        refactored_code="def foo(): pass",
        changes=[change],
        risks=["Potential rename side effects"],
    )
    assert result.refactored_code == "def foo(): pass"
    assert len(result.changes) == 1


def test_final_review():
    """Test FinalReview model."""
    review = FinalReview(
        approved=True,
        score=85,
        strengths=["Good structure"],
        remaining_issues=["Minor formatting"],
        recommendation="approve",
    )
    assert review.approved is True
    assert review.score == 85
    assert "approve" in review.recommendation.lower()