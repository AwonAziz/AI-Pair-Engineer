"""Command line interface.

The Streamlit app is the demo. This is the part that runs in CI, in a pre-commit
hook, or from a terminal when you do not want a browser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_pair_engineer.models.schemas import Severity
from ai_pair_engineer.pipeline import (
    SUPPORTED_LANGUAGES,
    PipelineResult,
    UnsupportedLanguageError,
    run_pipeline,
)
from ai_pair_engineer.services.llm import LLMError, MissingAPIKeyError
from ai_pair_engineer.static import analyze_source

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_FAILED = 4

SEVERITY_ORDER = {
    Severity.CRITICAL: "CRITICAL",
    Severity.HIGH: "HIGH",
    Severity.MEDIUM: "MEDIUM",
    Severity.LOW: "LOW",
}

_COLORS = {
    Severity.CRITICAL: "\033[41;97m",
    Severity.HIGH: "\033[91m",
    Severity.MEDIUM: "\033[93m",
    Severity.LOW: "\033[94m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _colorize(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{_RESET}" if enabled else text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pair-engineer",
        description="Multi-stage pre-review engineering partner.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="File to review. Omit with --stdin to read from standard input.",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        choices=SUPPORTED_LANGUAGES,
        help="Override language detection.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read the code to review from standard input.",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="text",
        choices=("text", "json", "markdown"),
        help="Output format.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override, for example anthropic/claude-sonnet-4.",
    )
    parser.add_argument(
        "--fail-on",
        default="high",
        choices=("critical", "high", "medium", "low", "never"),
        help=(
            "Exit non-zero when a finding at or above this severity exists. "
            "Defaults to high, which is the usual CI gate: critical alone lets "
            "serious defects through, and low alone fails on style preferences."
        ),
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run only local static analysis. No API key required.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")
    return parser


def _detect_language(path: Path | None, source: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if path is not None:
        suffix = path.suffix.lower().lstrip(".")
        if suffix in {"ts", "tsx"}:
            return "typescript"
        if suffix in {"js", "jsx", "mjs", "cjs"}:
            return "javascript"
        if suffix in {"py", "pyi"}:
            return "python"
        if suffix == "java":
            return "java"
    return "python"


class SourceReadError(ValueError):
    """The source to review could not be read from the requested location."""


def _read_source(args: argparse.Namespace) -> tuple[str, Path | None]:
    """Load the code under review from a path or from stdin."""
    if args.stdin:
        return sys.stdin.read(), None
    if not args.path:
        raise SourceReadError("usage: pair-engineer <file> | pair-engineer --stdin")
    path = Path(args.path)
    if not path.is_file():
        raise SourceReadError(f"no such file: {path}")
    return path.read_text(encoding="utf-8"), path


def _render_text(result: PipelineResult, *, color: bool) -> str:
    lines: list[str] = []
    counts = result.analysis.severity_counts()

    lines.append(_colorize(f"\n{result.analysis.summary}", _BOLD, color))
    lines.append("")

    for finding in result.analysis.findings:
        label = SEVERITY_ORDER[finding.severity]
        header = f"{label:8} {finding.category.value:16} {finding.title}"
        lines.append(_colorize(header, _COLORS[finding.severity], color))

        if finding.location:
            where = finding.location.render()
            if where:
                lines.append(_colorize(f"         at {where}", _DIM, color))

        lines.append(f"         {finding.description}")
        lines.append(_colorize(f"         fix: {finding.recommendation}", _DIM, color))
        lines.append("")

    summary = "  ".join(
        f"{SEVERITY_ORDER[severity]} {count}" for severity, count in counts.items() if count
    )
    lines.append(_colorize(f"Quality score {result.quality_score}/100", _BOLD, color))
    if summary:
        lines.append(_colorize(summary, _DIM, color))

    if result.tests.tests:
        lines.append("")
        lines.append(_colorize(f"Tests suggested ({len(result.tests.tests)})", _BOLD, color))
        for test in result.tests.tests:
            lines.append(f"  {test.name}")
            lines.append(_colorize(f"    {test.purpose}", _DIM, color))

    if result.refactor.changes:
        lines.append("")
        lines.append(_colorize(f"Refactor changes ({len(result.refactor.changes)})", _BOLD, color))
        for change in result.refactor.changes:
            lines.append(f"  {change.title}")
            lines.append(_colorize(f"    {change.rationale}", _DIM, color))

    for risk in result.refactor.risks:
        lines.append(_colorize(f"  risk: {risk}", _COLORS[Severity.MEDIUM], color))

    lines.append("")
    verdict = _colorize(result.review.verdict.upper(), _BOLD, color)
    lines.append(
        f"Verdict {verdict}  score {result.review.score}/100  "
        f"regression risk {result.review.regression_risk}"
    )
    lines.append(result.review.recommendation)

    for issue in result.review.remaining_issues:
        lines.append(_colorize(f"  - {issue}", _COLORS[Severity.HIGH], color))

    return "\n".join(lines)


def _render_markdown(result: PipelineResult, path: Path | None) -> str:
    """GitHub-flavoured Markdown, suitable for a PR comment."""
    title = f"## AI Pair Engineer: {path.name if path else 'review'}"
    lines = [title, ""]

    lines.append(
        f"**Quality score** {result.quality_score}/100 · **Verdict** `{result.review.verdict}`"
    )
    lines.append("")
    lines.append(result.analysis.summary)
    lines.append("")

    if not result.analysis.findings:
        lines.append("No findings.")
        return "\n".join(lines)

    lines.append("| | Severity | Category | Finding | Location |")
    lines.append("|---|---|---|---|---|")
    for finding in result.analysis.findings:
        location = finding.location.line if finding.location and finding.location.line else "—"
        lines.append(
            f"| {finding.title} | {finding.severity.value} | {finding.category.value} "
            f"| {finding.description} | {location} |"
        )

    lines.append("")
    for finding in result.analysis.findings:
        lines.append(f"**{finding.title}** ({finding.severity.value})")
        if finding.location:
            where = finding.location.render()
            if where:
                lines.append(f"- Location: {where}")
        lines.append(f"- Impact: {finding.description}")
        lines.append(f"- Fix: {finding.recommendation}")
        lines.append("")

    if result.tests.tests:
        lines.append("### Suggested tests")
        lines.append("")
        for test in result.tests.tests:
            lines.append(f"- `{test.name}` — {test.purpose}")
        lines.append("")

    if result.review.remaining_issues:
        lines.append("### Remaining issues")
        lines.append("")
        for issue in result.review.remaining_issues:
            lines.append(f"- {issue}")
        lines.append("")

    return "\n".join(lines)


def _exit_code(result: PipelineResult, fail_on: str) -> int:
    if fail_on == "never":
        return EXIT_OK
    threshold = Severity(fail_on)
    order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
    cutoff = order.index(threshold)
    for severity in order[: cutoff + 1]:
        if any(finding.severity == severity for finding in result.analysis.findings):
            return EXIT_FINDINGS
    return EXIT_OK


def _render_static(source: str, language: str, *, color: bool) -> str:
    report = analyze_source(source, language)
    lines = [f"{report.function_count} functions", f"{report.class_count} classes"]

    for metric in report.functions:
        notes: list[str] = [f"complexity {metric.cyclomatic_complexity}"]
        if metric.deeply_nested:
            notes.append(f"nesting {metric.max_nesting}")
        if metric.too_many_arguments:
            notes.append(f"{metric.argument_count} arguments")
        if metric.too_many_returns:
            notes.append(f"{metric.return_count} returns")
        if metric.docstring is None:
            notes.append("undocumented")
        lines.append(f"  {metric.name} (line {metric.lineno}): " + ", ".join(notes))

    for line_no in report.bare_except_lines:
        lines.append(_colorize(f"  bare except at line {line_no}", _COLORS[Severity.HIGH], color))
    for line_no in report.mutable_default_lines:
        lines.append(
            _colorize(f"  mutable default at line {line_no}", _COLORS[Severity.HIGH], color)
        )
    for name in report.unused_imports:
        lines.append(_colorize(f"  unused import: {name}", _COLORS[Severity.LOW], color))

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    color = not args.no_color and sys.stdout.isatty()

    try:
        source, path = _read_source(args)
    except SourceReadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(f"error: could not read the source file: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not source.strip():
        print("error: no source code to review", file=sys.stderr)
        return EXIT_USAGE

    language = _detect_language(path, source, args.language)

    if args.static_only:
        print(_render_static(source, language, color=color))
        return EXIT_OK

    try:
        result = run_pipeline(
            language,
            source,
            model=args.model,
        )
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: use --static-only to run local analysis without a key.",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    except UnsupportedLanguageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except LLMError as exc:
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return EXIT_FAILED

    if args.format == "json":
        print(json.dumps(result.to_dict(include_code=True), indent=2))
    elif args.format == "markdown":
        print(_render_markdown(result, path))
    else:
        print(_render_text(result, color=color))

    return _exit_code(result, args.fail_on)


if __name__ == "__main__":
    raise SystemExit(main())
