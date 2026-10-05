"""Deterministic, local static analysis for Python.

This runs before the LLM and produces evidence the model cannot cheaply derive
on its own: exact line numbers, real cyclomatic complexity, and verified facts
about the syntax tree. It is also the part of the system that works with no API
key, which makes it useful on its own.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

# Branch constructs that each add one path to a function's control flow.
_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.IfExp,
    ast.comprehension,
)

# Names that signal a function does more than one job.
_GOD_FUNCTION_THRESHOLD = 60
_DEEP_NESTING_THRESHOLD = 4
_HIGH_COMPLEXITY_THRESHOLD = 10


@dataclass(frozen=True)
class FunctionMetrics:
    name: str
    lineno: int
    end_lineno: int
    cyclomatic_complexity: int
    max_nesting: int
    argument_count: int
    return_count: int
    docstring: str | None

    @property
    def too_many_arguments(self) -> bool:
        return self.argument_count > 4

    @property
    def too_many_returns(self) -> bool:
        return self.return_count > 4

    @property
    def deeply_nested(self) -> bool:
        return self.max_nesting >= _DEEP_NESTING_THRESHOLD

    @property
    def complex(self) -> bool:
        return self.cyclomatic_complexity > _HIGH_COMPLEXITY_THRESHOLD


@dataclass(frozen=True)
class StaticReport:
    """Facts about a source file, computed without a model."""

    language: str
    supported: bool
    syntax_error: str | None = None
    function_count: int = 0
    class_count: int = 0
    max_nesting: int = 0
    total_nodes: int = 0
    untested_function_names: list[str] = field(default_factory=list)
    functions: list[FunctionMetrics] = field(default_factory=list)
    uses_bare_except: bool = False
    bare_except_lines: list[int] = field(default_factory=list)
    mutable_default_lines: list[int] = field(default_factory=list)
    unused_imports: list[str] = field(default_factory=list)
    long_lines: list[int] = field(default_factory=list)
    todo_comments: list[int] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.syntax_error
            or self.bare_except_lines
            or self.mutable_default_lines
            or self.unused_imports
            or any(
                metric.too_many_arguments
                or metric.too_many_returns
                or metric.deeply_nested
                or metric.complex
                or metric.docstring is None
                for metric in self.functions
            )
        )

    def summary_lines(self) -> list[str]:
        """Render the report as short prompt-ready lines.

        Only non-default facts are emitted, so a clean file produces a short
        context block instead of a wall of zeroes.
        """
        if not self.supported:
            return ["  static analysis: not available for this language"]

        lines = [
            f"  functions: {self.function_count}",
            f"  classes: {self.class_count}",
            f"  max nesting depth: {self.max_nesting}",
        ]

        if self.syntax_error:
            lines.append(f"  SYNTAX ERROR: {self.syntax_error}")
        if self.bare_except_lines:
            where = ", ".join(str(line) for line in self.bare_except_lines)
            lines.append(f"  bare `except:` at line(s) {where}")
        if self.mutable_default_lines:
            lines.append(
                "  mutable default argument at line(s) "
                f"{', '.join(map(str, self.mutable_default_lines))}"
            )
        if self.unused_imports:
            lines.append(f"  unused import(s): {', '.join(self.unused_imports)}")
        if self.long_lines:
            lines.append(f"  over-long line(s): {', '.join(map(str, self.long_lines))[:200]}")
        if self.todo_comments:
            lines.append(f"  TODO/FIXME at line(s) {', '.join(map(str, self.todo_comments))}")

        for metric in self.functions:
            notes: list[str] = []
            if metric.complex:
                notes.append(f"cyclomatic complexity {metric.cyclomatic_complexity}")
            if metric.deeply_nested:
                notes.append(f"nesting depth {metric.max_nesting}")
            if metric.too_many_arguments:
                notes.append(f"{metric.argument_count} arguments")
            if metric.too_many_returns:
                notes.append(f"{metric.return_count} return statements")
            if metric.docstring is None:
                notes.append("no docstring")
            if notes:
                lines.append(f"  `{metric.name}` (line {metric.lineno}): " + ", ".join(notes))

        return lines


def analyze_source(source_code: str, language: str) -> StaticReport:
    """Compute a deterministic report for ``source_code``.

    Python is analysed via the ``ast`` module. Other languages return a report
    with ``supported`` set to False, so the caller can degrade gracefully
    instead of pretending to have evidence.
    """
    if language.lower() != "python":
        return StaticReport(language=language, supported=False)

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return StaticReport(
            language=language,
            supported=True,
            syntax_error=f"line {exc.lineno}: {exc.msg}",
        )

    return _analyze_python(tree, source_code)


def _analyze_python(tree: ast.Module, source_code: str) -> StaticReport:
    body = _body_of(tree)

    # File-level nesting is the deepest chain found anywhere, counting both
    # top-level control flow and control flow inside each function. Taking only
    # the module body would report 0 for every file that wraps its logic in
    # functions, which is exactly where nesting is worth knowing about.
    functions = [
        _measure_function(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    top_level = _max_nesting(body)
    nested_in_function = max((metric.max_nesting for metric in functions), default=0)

    # Public functions are the ones a caller can reach; dunders and nested
    # helpers are excluded so the list stays actionable.
    public = [metric for metric in functions if not metric.name.startswith("_")]

    bare_except = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    )

    return StaticReport(
        language="python",
        supported=True,
        function_count=len(functions),
        class_count=sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)),
        max_nesting=max(top_level, nested_in_function),
        total_nodes=sum(1 for _ in ast.walk(tree)),
        untested_function_names=[metric.name for metric in public],
        functions=functions,
        uses_bare_except=bool(bare_except),
        bare_except_lines=bare_except,
        mutable_default_lines=_mutable_default_lines(tree, module_mutables(tree)),
        unused_imports=_unused_imports(tree),
        long_lines=_long_lines(source_code, limit=120),
        todo_comments=_todo_comment_lines(source_code),
    )


def _measure_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionMetrics:
    complexity = _cyclomatic_complexity(node)
    spec = node.args
    # vararg and kwarg are excluded on purpose: they signal a parameter bundle
    # rather than a wide interface, and counting them would make a tidy
    # signature look long.
    named_arguments = len(spec.posonlyargs) + len(spec.args) + len(spec.kwonlyargs)
    return FunctionMetrics(
        name=node.name,
        lineno=node.lineno,
        end_lineno=node.end_lineno or node.lineno,
        cyclomatic_complexity=complexity,
        max_nesting=_max_nesting(_body_of(node)),
        argument_count=named_arguments,
        return_count=sum(1 for child in ast.walk(node) if isinstance(child, ast.Return)),
        docstring=ast.get_docstring(node),
    )


def _cyclomatic_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """McCabe complexity: one plus every branch that adds an independent path.

    Nested function definitions are skipped so their branches are not double
    counted against the enclosing function.
    """
    complexity = 1
    stack: list[ast.AST] = list(_body_of(node))

    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(current, _BRANCH_NODES):
            complexity += 1
        if isinstance(current, ast.BoolOp):
            complexity += max(0, len(current.values) - 1)
        stack.extend(ast.iter_child_nodes(current))

    return complexity


def _body_of(node: ast.AST) -> list[ast.AST]:
    return list(getattr(node, "body", []))


def _max_nesting(body: list[ast.AST]) -> int:
    """Deepest chain of control-flow constructs, excluding function boundaries."""
    deepest = 0
    stack: list[tuple[list[ast.AST], int]] = [(body, 0)]

    while stack:
        nodes, depth = stack.pop()
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                deepest = max(deepest, depth)
                continue
            if isinstance(node, _BRANCH_NODES):
                deepest = max(deepest, depth + 1)
                stack.append((_body_of(node), depth + 1))
            else:
                stack.append((_body_of(node), depth))

    return deepest


def module_mutables(tree: ast.Module) -> set[str]:
    """Names bound at module level to a mutable literal or constructor.

    ``CACHE = {}`` followed by ``def clear(cache=CACHE)`` is the same shared-state
    bug as ``def clear(cache={})``, so the default-argument check needs to know
    which module-level names hold a mutable. Only direct literal and constructor
    bindings are followed; anything needing evaluation is skipped rather than
    guessed at.
    """
    mutables: set[str] = set()

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue

        for target in targets:
            if isinstance(target, ast.Name) and _is_mutable_expression(value):
                mutables.add(target.id)

    return mutables


def _is_mutable_expression(node: ast.expr) -> bool:
    if isinstance(node, (ast.List, ast.Dict, ast.Set)):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "dict", "set"}
    )


def _mutable_default_lines(tree: ast.Module, mutable_names: set[str] | None = None) -> list[int]:
    """Locate default arguments that are shared across every call.

    A mutable default is evaluated once at definition time, so state written
    through it persists between calls. That is the classic ``def f(x=[])``
    bug, and it is invisible at the call site.
    """
    known_mutables = mutable_names if mutable_names is not None else set()
    lines: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        defaults = [*node.args.defaults, *node.args.kw_defaults]
        for default in defaults:
            if default is None:
                continue
            if _is_mutable_expression(default):
                lines.append(default.lineno)
            elif isinstance(default, ast.Name) and default.id in known_mutables:
                # `def clear(cache=CACHE)` where CACHE is a module-level dict.
                lines.append(default.lineno)

    return sorted(set(lines))


def _unused_imports(tree: ast.Module) -> list[str]:
    """Names imported via a plain ``import x`` or ``from x import y`` and never referenced.

    Deliberately conservative: it skips star imports, aliased imports, and
    anything referenced inside a string annotation, because a false positive
    here would push the reviewer toward removing a needed import.
    """
    imported: dict[str, int] = {}
    star = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None and alias.name.count(".") > 1:
                    continue  # import a.b.c binds a, handled below
                if alias.asname is not None:
                    continue  # aliased: reference site is not obvious
                imported[alias.name.split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                star = True
                continue
            for alias in node.names:
                if alias.asname is not None:
                    continue
                imported[alias.name] = node.lineno

    if not imported or star:
        return []

    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(_attribute_root(node))

    return sorted(name for name in imported if name not in used)


def _attribute_root(node: ast.Attribute) -> str:
    """Return the base name of a dotted reference such as ``os.path.join``."""
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else ""


def _long_lines(source_code: str, *, limit: int) -> list[int]:
    return [
        number for number, line in enumerate(source_code.splitlines(), start=1) if len(line) > limit
    ]


def _todo_comment_lines(source_code: str) -> list[int]:
    markers = ("TODO", "FIXME", "XXX", "HACK")
    hits: list[int] = []
    for number, line in enumerate(source_code.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") and any(marker in stripped for marker in markers):
            hits.append(number)
    return hits


def to_evidence(report: StaticReport) -> dict[str, Any]:
    """Serialise a report for structured logging or a JSON CLI output."""
    return {
        "language": report.language,
        "supported": report.supported,
        "syntax_error": report.syntax_error,
        "function_count": report.function_count,
        "class_count": report.class_count,
        "max_nesting": report.max_nesting,
        "total_nodes": report.total_nodes,
        "bare_except_lines": report.bare_except_lines,
        "mutable_default_lines": report.mutable_default_lines,
        "unused_imports": report.unused_imports,
        "long_lines": report.long_lines,
        "todo_comments": report.todo_comments,
        "functions": [
            {
                "name": metric.name,
                "line": metric.lineno,
                "cyclomatic_complexity": metric.cyclomatic_complexity,
                "max_nesting": metric.max_nesting,
                "arguments": metric.argument_count,
                "returns": metric.return_count,
                "documented": metric.docstring is not None,
            }
            for metric in report.functions
        ],
    }
