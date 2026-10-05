"""Streamlit front end.

Kept deliberately thin: it collects input, calls the pipeline, and renders the
result. All review logic lives in the package so the CLI and the UI cannot
drift apart.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from ai_pair_engineer.models.schemas import Severity
from ai_pair_engineer.pipeline import (
    SUPPORTED_LANGUAGES,
    PipelineResult,
    UnsupportedLanguageError,
    run_pipeline,
)
from ai_pair_engineer.services.llm import LLMError, MissingAPIKeyError, is_configured
from ai_pair_engineer.static import analyze_source

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="AI Pair Engineer", page_icon="🔍", layout="wide")

LANGUAGE_LABELS = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
}

SEVERITY_ICONS = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
}

EXAMPLE = """def process_users(users):
    results = []
    for user in users:
        if user["age"] >= 18:
            if user["email"] != "":
                email = user["email"].strip().lower()
                if "@" in email:
                    results.append({"email": email, "adult": True})
    return results
"""


def _reset_results() -> None:
    st.session_state.pop("result", None)
    st.session_state.pop("static_only", None)


def render_sidebar() -> tuple[str, str, str]:
    """Draw the input controls and return (language, source, model)."""
    with st.sidebar:
        st.header("Input")

        language = st.selectbox(
            "Language",
            list(SUPPORTED_LANGUAGES),
            format_func=lambda value: LANGUAGE_LABELS[value],
        )

        model = st.text_input(
            "Model override",
            placeholder="leave blank to use $MODEL",
            help="Any OpenRouter model id, for example anthropic/claude-sonnet-4.",
        )

        st.divider()
        st.subheader("Source")
        uploaded = st.file_uploader("Upload a file", type=["py", "js", "ts", "java"])
        pasted = st.text_area(
            "or paste code",
            height=260,
            placeholder="def process(users): ...",
            key="source_input",
        )

        if uploaded is not None:
            source = uploaded.getvalue().decode("utf-8", errors="replace")
        else:
            source = pasted

        if st.button("Load example", use_container_width=True):
            st.session_state["source_input"] = EXAMPLE
            st.rerun()

    return language, source, model.strip()


def render_config_notice() -> None:
    """Explain the two ways to run: free local analysis, or full pipeline."""
    if is_configured():
        return

    st.warning(
        "No `OPENROUTER_API_KEY` found. Local static analysis still works below, "
        "but the four-stage review needs a key.",
        icon="⚠️",
    )


def render_static_panel(source: str, language: str) -> None:
    with st.expander("Local static analysis", expanded=True):
        st.caption(
            "Computed from the Python AST on this machine. No API calls, no key "
            "required, and the line numbers are exact."
        )
        report = analyze_source(source, language)
        if not report.supported:
            st.info(f"Static analysis is only implemented for Python, not {language}.")
            return
        if report.syntax_error:
            st.error(f"This file does not parse: {report.syntax_error}")
            return

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Functions", report.function_count)
        col2.metric("Classes", report.class_count)
        col3.metric("Max nesting", report.max_nesting)
        col4.metric("Nodes", report.total_nodes)

        if report.functions:
            st.markdown("**Per-function metrics**")
            st.dataframe(
                [
                    {
                        "function": metric.name,
                        "line": metric.lineno,
                        "complexity": metric.cyclomatic_complexity,
                        "nesting": metric.max_nesting,
                        "args": metric.argument_count,
                        "returns": metric.return_count,
                        "documented": metric.docstring is not None,
                    }
                    for metric in report.functions
                ],
                use_container_width=True,
                hide_index=True,
            )

        issues = (
            [
                (str(line), "Bare `except:` swallows every error")
                for line in report.bare_except_lines
            ]
            + [
                (str(line), "Mutable default argument is shared across calls")
                for line in report.mutable_default_lines
            ]
            + [(name, "Imported but never referenced") for name in report.unused_imports]
        )
        if issues:
            st.markdown("**Deterministic issues**")
            for location, description in issues:
                st.write(f"`{location}` — {description}")


def _progress(stage: str, status: str) -> None:
    st.session_state["current_stage"] = f"{stage}: {status}"
    st.caption(st.session_state["current_stage"])


def run_and_store(language: str, source: str, model: str | None) -> None:
    result: PipelineResult = run_pipeline(
        language,
        source,
        model=model or None,
        on_progress=_progress,
    )
    st.session_state["result"] = result


def render_overview(result: PipelineResult) -> None:
    counts = result.analysis.severity_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quality score", f"{result.quality_score}/100")
    col2.metric("Critical", counts[Severity.CRITICAL])
    col3.metric("High", counts[Severity.HIGH])
    col4.metric("Tests suggested", len(result.tests.tests))

    st.markdown(result.analysis.summary)

    if result.review.approved:
        st.success(f"Approved · score {result.review.score}/100", icon="✅")
    else:
        st.error(
            f"{result.review.verdict.replace('_', ' ').upper()} · score {result.review.score}/100",
            icon="⚠️",
        )

    if result.review.regression_risk != "low":
        st.warning(f"Regression risk assessed as {result.review.regression_risk}.")

    usage = result.trace.to_dict()["total_tokens"]
    if usage:
        st.caption(f"{usage} tokens across {len(result.trace.stages)} stages")


def render_findings(result: PipelineResult) -> None:
    if not result.analysis.findings:
        st.success("No findings. The analyzer found nothing worth reporting.")
        return

    grouped: dict[Severity, list[Any]] = {}
    for finding in result.analysis.findings:
        grouped.setdefault(finding.severity, []).append(finding)

    for severity in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
        items = grouped.get(severity)
        if not items:
            continue
        st.subheader(f"{SEVERITY_ICONS[severity]} {severity.value.title()} ({len(items)})")
        for finding in items:
            with st.expander(f"{finding.title} — `{finding.category.value}`"):
                if finding.location:
                    where = finding.location.render()
                    if where:
                        st.caption(where)
                st.write(finding.description)
                st.markdown(f"**Suggested fix:** {finding.recommendation}")
                st.caption(f"Model confidence: {finding.confidence:.0%}")


def render_tests(result: PipelineResult) -> None:
    if not result.tests.tests:
        st.info("No tests were generated.")
        return

    for test in result.tests.tests:
        with st.expander(f"`{test.name}`"):
            st.caption(test.purpose)
            if test.covers:
                st.caption("Covers: " + ", ".join(test.covers))
            st.code(test.test_code, language=result.language)


def render_refactor(result: PipelineResult) -> None:
    left, right = st.columns(2)

    with left:
        st.subheader("Original")
        st.code(result.source_code, language=result.language)
    with right:
        st.subheader("Refactored")
        st.code(result.refactor.refactored_code, language=result.language)

    if result.refactor.summary:
        st.markdown(result.refactor.summary)

    if result.refactor.changes:
        st.subheader("Changes")
        for change in result.refactor.changes:
            st.markdown(f"**{change.title}**")
            st.write(change.description)
            st.caption(f"Rationale: {change.rationale}")

    if result.refactor.risks:
        st.subheader("Risks")
        for risk in result.refactor.risks:
            st.warning(risk, icon="⚠️")


def render_review(result: PipelineResult) -> None:
    st.subheader(result.review.summary or "Final verdict")

    if result.review.approved:
        st.success("Approved", icon="✅")
    else:
        st.error(result.review.verdict.replace("_", " ").upper(), icon="⚠️")

    col1, col2 = st.columns(2)
    col1.metric("Reviewer score", f"{result.review.score}/100")
    col2.metric("Regression risk", result.review.regression_risk)

    if result.review.strengths:
        st.subheader("What improved")
        for strength in result.review.strengths:
            st.write(f"- {strength}")

    if result.review.remaining_issues:
        st.subheader("Still outstanding")
        for issue in result.review.remaining_issues:
            st.write(f"- {issue}")

    st.subheader("Recommendation")
    st.write(result.review.recommendation)


def render_results(result: PipelineResult) -> None:
    st.divider()
    tabs = st.tabs(["Overview", "Findings", "Tests", "Refactor", "Review", "Static"])

    with tabs[0]:
        render_overview(result)
    with tabs[1]:
        render_findings(result)
    with tabs[2]:
        render_tests(result)
    with tabs[3]:
        render_refactor(result)
    with tabs[4]:
        render_review(result)
    with tabs[5]:
        render_static_panel(result.source_code, result.language)


def main() -> None:
    st.title("AI Pair Engineer")
    st.caption(
        "Four-stage review before a human sees the pull request: analyze, test, "
        "refactor, then adversarially verify the refactor."
    )

    language, source, model = render_sidebar()
    render_config_notice()

    ready = bool(source and source.strip())
    if not ready:
        st.info("Paste some code, upload a file, or load the example to begin.")
        if source:
            render_static_panel(source, language)
        return

    if st.button("Run review", type="primary", use_container_width=True):
        _reset_results()
        try:
            with st.spinner("Running the four-stage pipeline..."):
                run_and_store(language, source, model)
        except MissingAPIKeyError as exc:
            st.error(str(exc), icon="🔑")
        except UnsupportedLanguageError as exc:
            st.error(str(exc), icon="🚫")
        except LLMError as exc:
            st.error(f"The review pipeline failed: {exc}", icon="⚠️")
        else:
            st.success("Review complete.")

    result = st.session_state.get("result")
    if isinstance(result, PipelineResult):
        render_results(result)


if __name__ == "__main__":
    main()
