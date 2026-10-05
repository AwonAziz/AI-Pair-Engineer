"""Command line entry point.

    python -m benchmarks validate
    python -m benchmarks run --tiers static
    python -m benchmarks run --tiers all --regressions --regressions-baseline
    python -m benchmarks run --tiers analyzer+static --model anthropic/claude-sonnet-4

``validate`` needs no API key and no network, so it runs in CI. Everything
under ``run`` that touches a model tier needs ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from benchmarks.corpus import CorpusError, load_cases, load_regressions
from benchmarks.models import RunReport, Tier
from benchmarks.report import (
    NOT_APPLICABLE,
    render_category_breakdown,
    render_json,
    render_markdown,
    render_severity_breakdown,
    tier_order,
)
from benchmarks.runner import run_benchmark
from benchmarks.scoring import (
    recall_by_case,
    recall_by_category,
    recall_by_severity,
    severity_weighted_recall,
)
from benchmarks.validate import validate_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks",
        description="Measure defect detection against a corpus of planted defects.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Check corpus annotations. No API key or network needed.",
    )
    validate.set_defaults(handler=_cmd_validate)

    run = subparsers.add_parser("run", help="Run the benchmark.")
    run.add_argument(
        "--tiers",
        default="static",
        help=(
            "Comma-separated tiers to compare: static, naive, analyzer, "
            "analyzer+static, or 'all'. Default: static."
        ),
    )
    run.add_argument("--model", default=None, help="Model id. Defaults to $MODEL.")
    run.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for every model call. Default 0 for comparability.",
    )
    run.add_argument(
        "--regressions",
        action="store_true",
        help="Also measure behaviour-change detection by the reviewer.",
    )
    run.add_argument(
        "--regressions-baseline",
        action="store_true",
        help="Use the plain reviewer prompt instead of the adversarial one.",
    )
    run.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case ids to restrict the run to.",
    )
    run.add_argument("--format", default="markdown", choices=("markdown", "json"))
    run.add_argument("--out", default=None, help="Write the report to a file.")
    run.add_argument(
        "--refresh", action="store_true", help="Ignore cached results and overwrite them."
    )
    run.add_argument(
        "--no-cache", action="store_true", help="Do not read or write the result cache."
    )
    run.add_argument(
        "--breakdown",
        action="store_true",
        help="Add per-category and per-severity recall tables.",
    )
    run.add_argument("--verbose", "-v", action="store_true", help="Print per-case detail.")
    run.set_defaults(handler=_cmd_run)

    return parser


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate corpus annotations. Cheap enough for CI."""
    problems, case_count, defect_count = validate_corpus()

    print(f"Checked {case_count} case(s) and {defect_count} planted defect(s).")

    regressions = load_regressions()
    print(f"Checked {len(regressions)} regression case(s).")

    if problems:
        print(f"\n{len(problems)} problem(s) found:\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nCorpus is consistent.")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    requested = [part.strip() for part in args.tiers.split(",") if part.strip()]
    tiers = tier_order(requested)

    unknown = set(requested) - {"all"} - {tier.value for tier in tiers}
    if unknown:
        print(f"error: unknown tier(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        print(
            "valid tiers: static, naive, analyzer, analyzer+static, all",
            file=sys.stderr,
        )
        return 2

    if not tiers:
        print("error: no valid tiers requested", file=sys.stderr)
        return 2

    needs_model = any(tier.uses_model for tier in tiers) or args.regressions
    if needs_model:
        from ai_pair_engineer.services.llm import is_configured

        if not is_configured():
            print(
                "error: OPENROUTER_API_KEY is not set.\n"
                "       Only the 'static' tier runs without it.",
                file=sys.stderr,
            )
            return 3

    case_filter = (
        [part.strip() for part in args.cases.split(",") if part.strip()] if args.cases else None
    )

    # A corpus problem must fail before any money is spent. The load itself can
    # also fail, so it is inside the try: a malformed manifest is the caller's
    # problem to fix, not a traceback.
    try:
        problems, _, _ = validate_corpus()
    except CorpusError as exc:
        print(f"error: corpus could not be loaded: {exc}", file=sys.stderr)
        return 2

    if problems:
        print("error: corpus annotations are invalid:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    try:
        report = run_benchmark(
            tiers,
            model=args.model,
            temperature=args.temperature,
            refresh=args.refresh,
            use_cache=not args.no_cache,
            run_regressions=args.regressions or args.regressions_baseline,
            naive_reviewer=args.regressions_baseline,
            case_filter=case_filter,
            verbose=args.verbose,
        )
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        rendered = render_json(report, verbose=args.verbose)
    else:
        rendered = render_markdown(report, verbose=args.verbose)
        if args.breakdown:
            # Rendered for a single tier too: a breakdown of one column is how
            # you find out which category the tier is blind to.
            rendered += "\n" + _render_breakdowns(report, tiers[0])

    if args.out:
        from pathlib import Path

        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(rendered)

    return 0


def _render_breakdowns(report: RunReport, tier: Tier) -> str:
    """Append per-category and per-severity recall tables for one tier."""
    cases = {case.id: case for case in load_cases()}

    pairs = [
        (cases[result.case_id], result)
        for result in report.case_results
        if result.tier is tier and result.case_id in cases
    ]

    if not pairs:
        return ""

    category = recall_by_category(pairs)
    severity = recall_by_severity(pairs)
    macro = recall_by_case(pairs)
    weighted = severity_weighted_recall(pairs)

    lines = ["## Breakdown", ""]
    lines.append(
        f"**{tier.label}** - macro recall {_percent(macro)}, "
        f"severity-weighted {_percent(weighted)}."
    )
    lines += ["", *render_category_breakdown(report, category), ""]
    lines += [*render_severity_breakdown(report, severity), ""]
    lines.append(
        "A tier that leads on complexity and trails on security is not a working "
        "reviewer, which is why these breakdowns exist."
    )
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    if value is None:
        return NOT_APPLICABLE
    return f"{value * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
