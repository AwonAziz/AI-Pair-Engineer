from __future__ import annotations

import os
import traceback
from typing import Any

import streamlit as st

from ai_pair_engineer.models.schemas import (
    AnalysisResult,
    FinalReview,
    Finding,
    TestCase,
    TestResult,
    RefactorResult,
)
from ai_pair_engineer.agents import analyze_code, generate_tests, refactor_code, review_refactor
from ai_pair_engineer.services.llm import ask_llm, parse_structured_response

# Page config
st.set_page_config(
    page_title="AI Pair Engineer",
    page_icon="🧑‍💻",
    layout="wide",
)


def initialize_session_state() -> None:
    """Reset all analysis state when user submits new code."""
    keys_to_keep = {"api_key_entered"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

    st.session_state.setdefault("analysis_done", False)
    st.session_state.setdefault("refactoring_done", False)
    st.session_state.setdefault("review_done", False)
    st.session_state.setdefault("api_key_entered", False)
    st.session_state.setdefault("language", "python")
    st.session_state.setdefault("source_code", "")
    st.session_state.setdefault("analysis_result", None)  # type: ignore[assignment]
    st.session_state.setdefault("test_result", None)  # type: ignore[assignment]
    st.session_state.setdefault("refactor_result", None)  # type: ignore[assignment]
    st.session_state.setdefault("review_result", None)  # type: ignore[assignment]


def validate_api_key() -> bool:
    """Check if API key is available.

    The LLM service validates the key at import time, so if the app
    starts, the key is assumed to be properly configured.
    """
    if st.session_state.get("api_key_entered"):
        return True

    # If we got here, the key must be set (LLM service validates at import)
    st.session_state["api_key_entered"] = True
    return True


def render_error(error_type: str, message: str) -> None:
    """Render a user-friendly error message."""
    st.error(f"{error_type}: {message}")
    logger_error = {
        "missing_key": logger.warning,
        "invalid_key": logger.warning,
        "timeout": logger.warning,
        "rate_limit": logger.warning,
        "parse_error": logger.warning,
        "invalid_input": logger.info,
    }
    logger_error.get(error_type, logger.warning)(message)


def run_analysis(language: str, source_code: str) -> None:
    """Run the multi-stage analysis pipeline."""
    try:
        # Stage 1: Code Analyzer
        with st.spinner("Stage 1: Code Analyzer..."):
            analysis_result = analyze_code(language, source_code)
        st.session_state.analysis_result = analysis_result
        st.session_state.analysis_done = True

        # Stage 2: Test Engineer
        with st.spinner("Stage 2: Test Engineer..."):
            test_result = generate_tests(language, source_code, analysis_result.findings)
        st.session_state.test_result = test_result
        st.session_state.refactoring_done = True  # flag so we know tests ran

        # Stage 3: Refactoring Engineer
        with st.spinner("Stage 3: Refactoring Engineer..."):
            refactor_result = refactor_code(language, source_code, analysis_result.findings)
        st.session_state.refactor_result = refactor_result
        st.session_state.refactoring_done = True

        # Stage 4: Final Reviewer
        with st.spinner("Stage 4: Final Reviewer..."):
            review_result = review_refactor(
                original_code=source_code,
                refactored_code=refactor_result.refactored_code,
                generated_tests=test_result.tests,
                analyzer_findings=analysis_result.findings,
            )
        st.session_state.review_result = review_result
        st.session_state.review_done = True

    except Exception as e:
        st.error(f"Analysis pipeline failed: {str(e)[:200]}")
        logger.error("Pipeline error: %s", str(e))


def main() -> None:
    initialize_session_state()

    # Header
    st.title("AI Pair Engineer")
    st.subheader("AI-assisted code review, testing and refactoring before human review.")

    # API key check
    if not validate_api_key():
        st.warning("Please configure OPENROUTER_API_KEY environment variable.")
        st.stop()

    # Language selection
    language = st.selectbox(
        "Programming Language",
        ["Python", "JavaScript", "TypeScript", "Java"],
        index=0,
        key="language_selector",
    )

    # Code input
    source_code = st.text_area(
        "Source Code",
        height=300,
        placeholder="Paste your code here...",
        key="code_editor",
    )

    # Analyze button
    if st.button("Analyze Code", type="primary", use_container_width=True):
        if not source_code.strip():
            st.error("Please enter source code to analyze.")
            return

        # Store the language and code
        st.session_state.language = language
        st.session_state.source_code = source_code

        # Run the pipeline
        with st.spinner("Running multi-stage analysis pipeline..."):
            run_analysis(language, source_code)
        st.rerun()

    # Display results if analysis is done
    if st.session_state.get("analysis_done"):
        display_results()


def display_results() -> None:
    """Display all analysis results in tabs."""
    analysis_result = st.session_state.analysis_result  # type: ignore[assignment]
    test_result = st.session_state.test_result  # type: ignore[assignment]
    refactor_result = st.session_state.refactor_result  # type: ignore[assignment]
    review_result = st.session_state.review_result  # type: ignore[assignment]

    # --- Overview Tab ---
    tab_overview, tab_analysis, tab_tests, tab_refactor, tab_review = st.tabs(
        ["Overview", "Code Analysis", "Tests", "Refactoring", "Final Review"],
    )

    with tab_overview:
        st.header("Project Overview")

        if analysis_result and analysis_result.findings:
            severity_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
            for f in analysis_result.findings:
                severity_counts[f.severity.value] = severity_counts.get(f.severity.value, 0) + 1

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Findings", len(analysis_result.findings))
            with col2:
                st.metric("Critical/High", severity_counts.get("high", 0) + severity_counts.get("critical", 0))
            with col3:
                st.metric("Suggested Tests", len(test_result.tests) if test_result else 0)

            # Overall score calculation
            weight = {"critical": 25, "high": 15, "medium": 8, "low": 3}
            deduct = sum(weight.get(f.severity.value, 0) for f in analysis_result.findings)
            score = max(0, 100 - deduct)
            st.metric("Quality Score", f"{score}/100")

        if refactor_result:
            st.metric("Refactoring Changes", len(refactor_result.changes))

        if review_result:
            status = "✅ Approved" if review_result.approved else "⚠️ Needs Review"
            st.metric("Recommendation", status)
            st.metric("Final Score", f"{review_result.score}/100")

    with tab_analysis:
        st.header("Code Analysis Findings")

        if analysis_result and analysis_result.findings:
            for i, finding in enumerate(analysis_result.findings, 1):
                with st.expander(f"#{i} {finding.title} [{finding.severity.value}]"):
                    st.write(f"**Category:** {finding.category.value}")
                    st.write(f"**Description:** {finding.description}")
                    st.write(f"**Recommendation:** {finding.recommendation}")
                    st.caption(f"**Confidence:** {finding.confidence:.0%}")
        else:
            st.info("No findings reported.")

    with tab_tests:
        st.header("Suggested Tests")

        if test_result and test_result.tests:
            st.write(f"Generated {len(test_result.tests)} test case(s):")
            for i, test in enumerate(test_result.tests, 1):
                with st.expander(f"Test #{i}: {test.name}"):
                    st.code(test.test_code, language=language.lower())
        else:
            st.info("No tests generated.")

        with st.expander("View test purposes"):
            if test_result and test_result.tests:
                for test in test_result.tests:
                    st.write(f"- **{test.name}**: {test.purpose}")

    with tab_refactor:
        st.header("Refactoring")

        if refactor_result:
            st.subheader("Summary")
            st.write(refactor_result.summary)

            st.subheader("Changes")
            if refactor_result.changes:
                for i, change in enumerate(refactor_result.changes, 1):
                    st.write(f"**{i}. {change.title}**")
                    st.write(f"*{change.description}*")
                    st.caption(f"Rationale: {change.rationale}")

            st.subheader("Risks")
            if refactor_result.risks:
                for risk in refactor_result.risks:
                    st.warning(risk)

            col1, col2 = st.columns(2)
            with col1:
                st.code(refactor_result.refactored_code, language=language.lower())
            with col2:
                st.code(st.session_state.source_code, language=language.lower())

        else:
            st.info("No refactoring performed.")

    with tab_review:
        st.header("Final Review")

        if review_result:
            # Approval status
            if review_result.approved:
                st.success("✅ **APPROVED**")
            else:
                st.error("⚠️ **NEEDS REVIEW**")

            st.header("Score")
            st.write(f"**{review_result.score}/100**")

            st.header("Strengths")
            if review_result.strengths:
                for strength in review_result.strengths:
                    st.success(f"• {strength}")
            else:
                st.info("No specific strengths identified.")

            st.header("Remaining Issues")
            if review_result.remaining_issues:
                for issue in review_result.remaining_issues:
                    st.warning(f"• {issue}")
            else:
                st.info("No remaining issues identified.")

            st.header("Recommendation")
            st.write(review_result.recommendation)

        else:
            st.info("No review performed.")


if __name__ == "__main__":
    import os
    main()