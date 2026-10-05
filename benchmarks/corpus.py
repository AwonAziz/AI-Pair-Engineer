"""Corpus loading.

Each case is a directory under ``benchmarks/cases``:

    cases/insecure_sql/
        source.py     the file to review
        case.toml     the defects planted in it

Splitting the source from its annotation keeps the code under review
readable on its own. It is a real file, linted and parsed like any other, so a
case cannot rot into something that no longer compiles.

``case.toml``:

    id = "insecure_sql"
    intent = "SQL injection via string formatting"
    tags = ["security", "database"]
    clean = false

    [[defects]]
    id = "sql-injection"
    summary = "Query built with an f-string"
    category = "security"
    severity = "critical"
    line = 14
    note = "Use a parameterised query."
    keywords = [["sql", "query"], ["f-string", "format", "interpolat", "concat"]]

Keywords are groups: every group must match somewhere in a finding's text.
Splitting them into several single-word groups would make the annotation
slack enough to credit the wrong finding.

Line numbers are the fragile part. A loader check in ``tests/test_corpus.py``
verifies that every declared line exists in the source and carries a non-empty
``keywords`` entry, so a bad annotation fails CI rather than quietly
inflating recall.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ai_pair_engineer.models.schemas import Category, Severity
from benchmarks.models import DetectionCase, PlantedDefect, RegressionCase

CASES_DIR = Path(__file__).resolve().parent / "cases"
REGRESSIONS_DIR = CASES_DIR / "regressions"


class CorpusError(RuntimeError):
    """The corpus on disk is malformed."""


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise CorpusError(f"missing manifest: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CorpusError(f"invalid TOML in {path}: {exc}") from exc


def _parse_defects(raw: list[dict[str, Any]], *, case_id: str) -> tuple[PlantedDefect, ...]:
    defects: list[PlantedDefect] = []

    for entry in raw:
        missing = {"id", "summary", "category", "severity", "line"} - entry.keys()
        if missing:
            raise CorpusError(
                f"case {case_id!r} defect {entry.get('id', '<unnamed>')!r} is missing "
                f"required field(s): {', '.join(sorted(missing))}"
            )

        try:
            category = Category(entry["category"])
        except ValueError as exc:
            raise CorpusError(
                f"case {case_id!r} defect {entry['id']!r} has unknown category "
                f"{entry['category']!r}; expected one of "
                f"{', '.join(c.value for c in Category)}"
            ) from exc

        try:
            severity = Severity(entry["severity"])
        except ValueError as exc:
            raise CorpusError(
                f"case {case_id!r} defect {entry['id']!r} has unknown severity "
                f"{entry['severity']!r}; expected one of "
                f"{', '.join(s.value for s in Severity)}"
            ) from exc

        keywords = tuple(tuple(str(word) for word in group) for group in entry.get("keywords", []))

        defects.append(
            PlantedDefect(
                id=str(entry["id"]),
                summary=str(entry["summary"]),
                category=category,
                severity=severity,
                line=int(entry["line"]),
                end_line=(int(entry["end_line"]) if entry.get("end_line") is not None else None),
                keywords=keywords,
                note=str(entry.get("note", "")),
            )
        )

    return tuple(defects)


def load_case(directory: Path) -> DetectionCase:
    """Load one detection case from its directory."""
    manifest_path = directory / "case.toml"
    source_path = directory / "source.py"

    if not source_path.is_file():
        raise CorpusError(f"case directory {directory.name!r} has no source.py")

    manifest = _load_toml(manifest_path)
    case_id = str(manifest.get("id", directory.name))
    defects = _parse_defects(manifest.get("defects", []), case_id=case_id)
    clean = bool(manifest.get("clean", False))

    if clean and defects:
        raise CorpusError(f"case {case_id!r} is marked clean but declares {len(defects)} defect(s)")

    return DetectionCase(
        id=case_id,
        language=str(manifest.get("language", "python")),
        source=source_path.read_text(encoding="utf-8"),
        defects=defects,
        intent=str(manifest.get("intent", "")),
        tags=tuple(str(tag) for tag in manifest.get("tags", [])),
        clean=clean,
    )


def load_cases(directory: Path | None = None) -> list[DetectionCase]:
    """Load every detection case, sorted by id for reproducible runs."""
    root = directory or CASES_DIR
    if not root.is_dir():
        raise CorpusError(f"corpus directory not found: {root}")

    cases: list[DetectionCase] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "regressions":
            continue
        if (child / "case.toml").is_file():
            cases.append(load_case(child))

    if not cases:
        raise CorpusError(f"no cases found under {root}")
    return cases


def load_regressions(directory: Path | None = None) -> list[RegressionCase]:
    """Load every regression case.

    A regression case is two files plus a manifest: ``original.py`` and
    ``refactored.py`` differing by exactly one behavioural change.
    """
    root = directory or REGRESSIONS_DIR
    if not root.is_dir():
        return []

    cases: list[RegressionCase] = []
    for child in sorted(root.iterdir()):
        manifest_path = child / "case.toml"
        if not child.is_dir() or not manifest_path.is_file():
            continue

        manifest = _load_toml(manifest_path)
        case_id = str(manifest.get("id", child.name))

        for required in ("original.py", "refactored.py"):
            if not (child / required).is_file():
                raise CorpusError(f"regression case {case_id!r} is missing {required}")

        expected = manifest.get("expected", [])
        if not expected:
            raise CorpusError(
                f"regression case {case_id!r} declares no `expected` phrases; without "
                "them a rejection cannot be checked for substance"
            )

        cases.append(
            RegressionCase(
                id=case_id,
                original=(child / "original.py").read_text(encoding="utf-8"),
                refactored=(child / "refactored.py").read_text(encoding="utf-8"),
                change=str(manifest.get("change", "")),
                expected=tuple(str(phrase) for phrase in expected),
                language=str(manifest.get("language", "python")),
                tags=tuple(str(tag) for tag in manifest.get("tags", [])),
            )
        )

    return cases


def validate_case(case: DetectionCase) -> list[str]:
    """Return a list of annotation problems with ``case``.

    Called by the test suite so a corpus regression fails CI. An empty list
    means the case is internally consistent.
    """
    problems: list[str] = []
    source_lines = case.source.splitlines()

    if not source_lines:
        problems.append("source is empty")

    for defect in case.defects:
        if defect.line < 1 or defect.line > len(source_lines):
            problems.append(
                f"defect {defect.id!r} declares line {defect.line}, outside the "
                f"{len(source_lines)}-line source"
            )
            continue

        if defect.end_line is not None and not (
            defect.line <= defect.end_line <= len(source_lines)
        ):
            problems.append(
                f"defect {defect.id!r} declares an invalid span {defect.line}-{defect.end_line}"
            )

        if not defect.keywords:
            problems.append(
                f"defect {defect.id!r} has no keywords, so it can only ever be "
                "credited by an exact line match"
            )
        elif any(not group for group in defect.keywords):
            problems.append(f"defect {defect.id!r} has an empty keyword group")

    return problems
