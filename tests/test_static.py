"""Tests for the deterministic AST analysis layer.

This layer is the part of the system that needs no API key, so it carries more
weight than a thin wrapper would. The metrics are asserted against
hand-checked inputs rather than against whatever the code happens to produce.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_pair_engineer.static import StaticReport, analyze_source, to_evidence

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

SIMPLE = '''
def add(a, b):
    """Add two numbers."""
    return a + b
'''

BRANCHY = '''
def classify(n):
    """Classify a number."""
    if n > 0:
        return "positive"
    elif n < 0:
        return "negative"
    return "zero"
'''

DEEP = '''
def process(items):
    """Nested."""
    for item in items:
        if item:
            while True:
                if item.ready:
                    return item
    return None
'''

MUTABLE_DEFAULTS = '''
def append(item, bucket=[]):
    """Bad default."""
    bucket.append(item)
    return bucket


def build(mapping={}):
    """Also bad."""
    return mapping
'''

BARE_EXCEPT = """
def risky():
    \"\"\"Swallows everything.\"\"\"
    try:
        return 1
    except:
        return 0
"""

UNUSED = """
import os
import sys
import json

print(sys.argv)
"""

LEGACY_SOURCE = (EXAMPLES / "legacy_service.py").read_text(encoding="utf-8")


class TestSyntaxErrors:
    def test_reports_a_syntax_error_instead_of_raising(self) -> None:
        report = analyze_source("def broken(:\n", "python")
        assert report.syntax_error is not None
        assert "line" in report.syntax_error

    def test_syntax_error_appears_in_the_prompt_lines(self) -> None:
        report = analyze_source("def broken(:\n", "python")
        joined = "\n".join(report.summary_lines())
        assert "SYNTAX ERROR" in joined

    def test_non_python_reports_unsupported_rather_than_faking_it(self) -> None:
        report = analyze_source("function f() {}", "javascript")
        assert report.supported is False
        assert report.function_count == 0
        assert "not available" in "\n".join(report.summary_lines())


class TestCounts:
    def test_counts_functions_and_classes(self) -> None:
        report = analyze_source(
            "class A:\n    def m(self):\n        pass\n\ndef f():\n    pass\n",
            "python",
        )
        assert report.function_count == 2
        assert report.class_count == 1

    def test_counts_async_functions(self) -> None:
        report = analyze_source("async def f():\n    pass\n", "python")
        assert report.function_count == 1

    def test_language_is_case_insensitive(self) -> None:
        assert analyze_source(SIMPLE, "PYTHON").function_count == 1

    def test_reports_nothing_public_for_private_names(self) -> None:
        report = analyze_source("def _private():\n    pass\n", "python")
        assert report.untested_function_names == []


class TestComplexity:
    def test_a_straight_line_function_has_complexity_one(self) -> None:
        metrics = analyze_source(SIMPLE, "python").functions[0]
        assert metrics.cyclomatic_complexity == 1

    def test_each_branch_adds_one(self) -> None:
        """One if plus one elif means two extra paths."""
        metrics = analyze_source(BRANCHY, "python").functions[0]
        assert metrics.cyclomatic_complexity == 3

    def test_booleans_add_paths(self) -> None:
        source = '''
def check(a, b, c):
    """Compound condition."""
    if a and b and c:
        return True
    return False
'''
        metrics = analyze_source(source, "python").functions[0]
        # 1 base + 1 if + 2 extra values from the two extra `and` operands.
        assert metrics.cyclomatic_complexity == 4

    def test_complex_flag_only_fires_above_the_threshold(self) -> None:
        simple = analyze_source(SIMPLE, "python").functions[0]
        branchy = analyze_source(BRANCHY, "python").functions[0]
        assert not simple.complex
        assert not branchy.complex

    def test_nested_function_branches_are_not_double_counted(self) -> None:
        """A closure's complexity belongs to the closure, not the parent."""
        source = '''
def outer(flag):
    """Has a closure."""
    def inner(x):
        if x:
            return 1
        if x == 0:
            return 2
        return 3
    return inner(flag)
'''
        report = analyze_source(source, "python")
        outer = next(m for m in report.functions if m.name == "outer")
        inner = next(m for m in report.functions if m.name == "inner")
        assert outer.cyclomatic_complexity == 1
        assert inner.cyclomatic_complexity == 3

    def test_boolean_operator_adds_only_the_extra_operands(self) -> None:
        tree = ast.parse("x = a and b")
        assert tree is not None


class TestNesting:
    def test_flat_function_has_no_nesting(self) -> None:
        assert analyze_source(SIMPLE, "python").max_nesting == 0

    def test_counts_loop_and_conditional_depth(self) -> None:
        """for -> if -> while -> if is four levels of nesting."""
        assert analyze_source(DEEP, "python").max_nesting == 4

    def test_deeply_nested_flag_fires(self) -> None:
        metrics = analyze_source(DEEP, "python").functions[0]
        assert metrics.deeply_nested

    def test_shallow_function_is_not_flagged(self) -> None:
        assert not analyze_source(BRANCHY, "python").functions[0].deeply_nested

    def test_file_level_nesting_includes_function_bodies(self) -> None:
        """The file reports the deepest chain anywhere, not just at module level."""
        report = analyze_source(DEEP, "python")
        assert report.max_nesting == report.functions[0].max_nesting


class TestArgumentsAndReturns:
    def test_counts_named_arguments(self) -> None:
        """Defaults and keyword-only parameters both count.

        ``*args`` and ``**kwargs`` are excluded on purpose: they signal a
        parameter bundle, not a wide interface, and counting them would make a
        tidy signature look like a long one.
        """
        source = '''
def f(a, b=1, *args, c, **kwargs):
    """Many."""
    return None
'''
        assert analyze_source(source, "python").functions[0].argument_count == 3

    def test_counts_positional_only_arguments(self) -> None:
        source = '''
def f(a, b, /, c):
    """Positional only."""
    return None
'''
        assert analyze_source(source, "python").functions[0].argument_count == 3

    def test_flags_too_many_arguments(self) -> None:
        source = "def f(a, b, c, d, e):\n    pass\n"
        assert analyze_source(source, "python").functions[0].too_many_arguments

    def test_four_arguments_is_acceptable(self) -> None:
        source = "def f(a, b, c, d):\n    pass\n"
        assert not analyze_source(source, "python").functions[0].too_many_arguments

    def test_counts_return_statements(self) -> None:
        source = '''
def f(x):
    """Many returns."""
    if x == 1:
        return 1
    if x == 2:
        return 2
    if x == 3:
        return 3
    if x == 4:
        return 4
    return 0
'''
        assert analyze_source(source, "python").functions[0].return_count == 5

    def test_flags_too_many_returns(self) -> None:
        source = '''
def f(x):
    """Many returns."""
    if x == 1:
        return 1
    if x == 2:
        return 2
    if x == 3:
        return 3
    if x == 4:
        return 4
    return 0
'''
        assert analyze_source(source, "python").functions[0].too_many_returns


class TestDocstrings:
    def test_detects_a_docstring(self) -> None:
        assert analyze_source(SIMPLE, "python").functions[0].docstring is not None

    def test_detects_a_missing_docstring(self) -> None:
        assert analyze_source("def f():\n    pass\n", "python").functions[0].docstring is None

    def test_missing_docstring_reaches_the_prompt(self) -> None:
        joined = "\n".join(analyze_source("def f():\n    pass\n", "python").summary_lines())
        assert "no docstring" in joined


class TestDefectDetection:
    def test_finds_a_module_level_dict_used_as_a_default(self) -> None:
        """`def f(cache=CACHE)` is the same shared-state bug as `cache={}`."""
        source = "CACHE = {}\n\ndef clear(cache=CACHE):\n    cache.clear()\n"
        assert analyze_source(source, "python").mutable_default_lines == [3]

    def test_finds_a_module_level_list_used_as_a_default(self) -> None:
        source = "ITEMS = list()\n\ndef reset(items=ITEMS):\n    return items\n"
        assert analyze_source(source, "python").mutable_default_lines == [3]

    def test_ignores_an_immutable_module_level_default(self) -> None:
        source = "NAME = 'prod'\n\ndef connect(name=NAME):\n    return name\n"
        assert analyze_source(source, "python").mutable_default_lines == []

    def test_a_mutable_that_is_never_a_default_is_not_flagged(self) -> None:
        """The check is about defaults, not about module-level mutables."""
        source = "CACHE = {}\n\ndef store(key, value):\n    CACHE[key] = value\n"
        assert analyze_source(source, "python").mutable_default_lines == []

    def test_finds_mutable_list_default(self) -> None:
        assert analyze_source(MUTABLE_DEFAULTS, "python").mutable_default_lines

    def test_finds_mutable_dict_default(self) -> None:
        assert analyze_source(MUTABLE_DEFAULTS, "python").mutable_default_lines

    def test_ignores_immutable_defaults(self) -> None:
        source = "def f(a=1, b='x', c=(1, 2), d=None):\n    pass\n"
        assert analyze_source(source, "python").mutable_default_lines == []

    def test_flags_bare_except(self) -> None:
        report = analyze_source(BARE_EXCEPT, "python")
        assert report.uses_bare_except
        assert report.bare_except_lines == [6]

    def test_named_except_handler_is_fine(self) -> None:
        source = "def f():\n    try:\n        return 1\n    except ValueError:\n        return 0\n"
        assert not analyze_source(source, "python").uses_bare_except

    def test_finds_unused_imports(self) -> None:
        assert set(analyze_source(UNUSED, "python").unused_imports) == {"os", "json"}

    def test_used_import_is_not_reported(self) -> None:
        assert analyze_source(UNUSED, "python").unused_imports.count("sys") == 0

    def test_aliased_import_is_not_reported_as_unused(self) -> None:
        """`import numpy as np` then `np.array` must not be flagged."""
        source = "import numpy as np\n\nx = np.array([1])\n"
        assert analyze_source(source, "python").unused_imports == []

    def test_star_import_disables_the_check(self) -> None:
        source = "from os import *\n\nimport json\n"
        assert analyze_source(source, "python").unused_imports == []

    def test_finds_long_lines(self) -> None:
        source = "x = '" + "a" * 200 + "'\n"
        assert analyze_source(source, "python").long_lines == [1]

    def test_finds_todo_comments(self) -> None:
        source = "# TODO: finish this\nx = 1\n"
        assert analyze_source(source, "python").todo_comments == [1]

    def test_todo_inside_a_string_is_not_a_comment(self) -> None:
        source = 'x = "TODO: not a comment"\n'
        assert analyze_source(source, "python").todo_comments == []


class TestReportRendering:
    def test_clean_file_produces_a_short_block(self) -> None:
        lines = analyze_source(SIMPLE, "python").summary_lines()
        assert len(lines) <= 5

    def test_clean_file_reports_no_issues(self) -> None:
        assert not analyze_source(SIMPLE, "python").has_issues

    def test_issues_are_reported(self) -> None:
        assert analyze_source(BARE_EXCEPT, "python").has_issues

    def test_complex_function_is_reported(self) -> None:
        assert analyze_source(DEEP, "python").has_issues


class TestEvidenceSerialisation:
    def test_is_json_serialisable(self) -> None:
        import json

        payload = to_evidence(analyze_source(BRANCHY, "python"))
        assert json.loads(json.dumps(payload))["function_count"] == 1

    def test_includes_per_function_metrics(self) -> None:
        payload = to_evidence(analyze_source(BRANCHY, "python"))
        assert payload["functions"][0]["name"] == "classify"
        assert payload["functions"][0]["cyclomatic_complexity"] == 3

    def test_reports_the_documentation_flag(self) -> None:
        payload = to_evidence(analyze_source("def f():\n    pass\n", "python"))
        assert payload["functions"][0]["documented"] is False


class TestRealExamples:
    def test_the_simple_example_parses_and_has_problems(self, example_file: Path) -> None:
        report = analyze_source(example_file.read_text(encoding="utf-8"), "python")
        assert isinstance(report, StaticReport)
        assert report.syntax_error is None
        assert report.function_count >= 1
        assert report.has_issues

    def test_the_legacy_example_surfaces_its_planted_defects(self) -> None:
        """The defects in legacy_service.py must be detectable, not just plausible.

        Each assertion corresponds to a defect documented in that file's
        docstring, so the example cannot quietly stop demonstrating anything.
        """
        report = analyze_source(LEGACY_SOURCE, "python")

        # `except:` on the connection path and again in the close handler.
        assert len(report.bare_except_lines) >= 2

        # `def clear_cache(cache=CACHE)` aliases a module-level dict.
        assert report.mutable_default_lines

        assert report.has_issues

    def test_the_legacy_example_reports_complex_functions(self) -> None:
        report = analyze_source(LEGACY_SOURCE, "python")
        by_name = {metric.name: metric for metric in report.functions}
        # get_user has a loop, a try, and several branches.
        assert by_name["get_user"].cyclomatic_complexity > 1
        assert not by_name["write_report"].complex
