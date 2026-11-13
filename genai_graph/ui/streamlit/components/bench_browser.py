"""Reusable Benchmark Outcomes and Evaluation Browser component for Streamlit.

Visual counterpart to ``cli bench tui`` and ``cli bench questions``:
- KPI summary metrics (total questions, strict accuracy, weighted accuracy, groundedness, tool calls)
- Filterable questions browser (status pills, document filter, text search)
- Side-by-side or stacked question inspection (Question, Gold Reference, Agent Answer, Judge Rationale)
- Execution trace & trajectory viewer (step-by-step tool calls, arguments, outputs, DecisionRAG navigation packs)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from genai_graph.bench.config import BenchConfig, load_bench_profile
from genai_graph.ui.textual.apps.bench_tui import BenchQuestionDetail, load_bench_dataset_with_results

# ---------------------------------------------------------------------------
# Profile Discovery
# ---------------------------------------------------------------------------


def discover_bench_profiles(config_path: Path | str | None = None) -> list[str]:
    """Discover available benchmark profiles from bench.yaml or global config."""
    profiles: list[str] = []
    p = Path(config_path) if config_path else Path("config/bench.yaml")
    if p.exists():
        try:
            import yaml

            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                p_dict = data.get("profiles", {})
                if isinstance(p_dict, dict):
                    profiles.extend(p_dict.keys())
        except Exception:
            pass

    if not profiles:
        profiles = ["default"]
    return sorted(set(profiles))


# ---------------------------------------------------------------------------
# Metrics and KPI Computation
# ---------------------------------------------------------------------------


def _compute_bench_metrics(items: list[BenchQuestionDetail]) -> dict[str, Any]:
    """Compute high-level accuracy and execution metrics from question details."""
    total = len(items)
    if total == 0:
        return {
            "total": 0,
            "evaluated": 0,
            "correct": 0,
            "partial": 0,
            "incorrect": 0,
            "unrun": 0,
            "strict_acc": 0.0,
            "weighted_acc": 0.0,
            "groundedness": 0.0,
            "avg_tools": 0.0,
            "avg_tokens": 0.0,
        }

    evaluated = sum(1 for q in items if q.has_score)
    correct = sum(1 for q in items if q.correctness == "correct")
    partial = sum(1 for q in items if q.correctness == "partial")
    incorrect = sum(1 for q in items if q.correctness == "incorrect")
    unrun = sum(1 for q in items if not q.has_run)

    # Scored subset metrics
    denom = max(1, evaluated)
    strict_acc = (correct / denom) * 100.0
    weighted_acc = ((correct + 0.5 * partial) / denom) * 100.0

    grounded = sum(1 for q in items if q.groundedness == "grounded")
    groundedness = (grounded / denom) * 100.0

    total_tools = sum(q.n_tool_calls for q in items if q.has_run)
    run_count = sum(1 for q in items if q.has_run)
    avg_tools = total_tools / max(1, run_count)

    total_tokens = sum(q.input_tokens + q.output_tokens for q in items if q.has_run)
    avg_tokens = total_tokens / max(1, run_count)

    return {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "unrun": unrun,
        "strict_acc": strict_acc,
        "weighted_acc": weighted_acc,
        "groundedness": groundedness,
        "avg_tools": avg_tools,
        "avg_tokens": avg_tokens,
    }


# ---------------------------------------------------------------------------
# Main Reusable Benchmark Browser Component
# ---------------------------------------------------------------------------


def render_bench_browser(
    cfg: BenchConfig | None = None,
    *,
    profile_name: str | None = None,
    config_path: Path | str | None = None,
) -> None:
    """Render the complete Benchmark Outcomes dashboard.

    Args:
        cfg: Optional pre-loaded BenchConfig instance.
        profile_name: Optional profile name override.
        config_path: Optional path to bench YAML configuration file.
    """
    # 1. Profile selection
    profiles = discover_bench_profiles(config_path)
    selected_profile = profile_name
    if not cfg:
        if len(profiles) > 1:
            default_p_idx = 0
            if "decision_fast" in profiles:
                default_p_idx = profiles.index("decision_fast")
            selected_profile = st.sidebar.selectbox(
                "Benchmark Profile",
                options=profiles,
                index=default_p_idx,
                help="Select benchmark execution and evaluation profile",
            )
        else:
            selected_profile = profiles[0]

        cfg_p = Path(config_path) if config_path else None
        try:
            cfg = load_bench_profile(profile_name=selected_profile, config_path=cfg_p)
        except Exception as exc:
            st.error(f"Failed to load benchmark profile '{selected_profile}': {exc}")
            return

    # 2. Load dataset with results
    with st.spinner(f"Loading results for profile '{cfg.profile_name}'..."):
        try:
            all_questions = load_bench_dataset_with_results(cfg)
        except Exception as exc:
            st.error(f"Error loading benchmark questions and results: {exc}")
            return

    if not all_questions:
        st.warning(f"No benchmark questions found for profile '{cfg.profile_name}'. Run 'cli bench run' first.")
        return

    # 3. KPI Metrics Cards
    metrics = _compute_bench_metrics(all_questions)

    st.markdown(f"### 📊 Benchmark Outcomes: `{cfg.profile_name}`")
    st.caption(
        f"Agent: `{cfg.agent_llm or 'default'}` · Judge: `{cfg.grader.llm or 'default'}` · Total Questions: `{metrics['total']}`"
    )

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric(
        "Strict Accuracy",
        f"{metrics['strict_acc']:.1f}%",
        f"{metrics['correct']}/{metrics['evaluated']} correct",
    )
    kpi2.metric(
        "Weighted Accuracy",
        f"{metrics['weighted_acc']:.1f}%",
        f"+{metrics['partial']} partial",
    )
    kpi3.metric(
        "Groundedness",
        f"{metrics['groundedness']:.1f}%",
        "factually cited",
    )
    kpi4.metric(
        "Avg Tool Calls",
        f"{metrics['avg_tools']:.1f}",
        "steps per answer",
    )
    kpi5.metric(
        "Evaluated Questions",
        f"{metrics['evaluated']}/{metrics['total']}",
        f"{metrics['unrun']} unrun" if metrics["unrun"] > 0 else "100% complete",
    )

    st.divider()

    # 4. Filters Bar
    col_status, col_doc, col_search = st.columns([3, 3, 4])

    with col_status:
        status_filter = st.selectbox(
            "Verdict Filter",
            options=["All", "✓ Correct", "~ Partial", "✗ Incorrect", "Unrun"],
            index=0,
            help="Filter questions by LLM judge correctness outcome",
        )

    all_docs = sorted({q.doc_name for q in all_questions if q.doc_name})
    with col_doc:
        doc_filter = st.selectbox(
            "Document Filter",
            options=["All Documents", *all_docs],
            index=0,
            help="Filter questions by source manual / technical order",
        )

    with col_search:
        search_query = st.text_input(
            "Search Questions / Answers",
            placeholder="Search keywords (e.g. servicing, safety, fuel)...",
            help="Filter questions by text in question, gold answer, or agent answer",
        )

    # Apply filters
    filtered = all_questions
    if status_filter == "✓ Correct":
        filtered = [q for q in filtered if q.correctness == "correct"]
    elif status_filter == "~ Partial":
        filtered = [q for q in filtered if q.correctness == "partial"]
    elif status_filter == "✗ Incorrect":
        filtered = [q for q in filtered if q.correctness == "incorrect"]
    elif status_filter == "Unrun":
        filtered = [q for q in filtered if not q.has_run]

    if doc_filter != "All Documents":
        filtered = [q for q in filtered if q.doc_name == doc_filter]

    if search_query.strip():
        sq = search_query.strip().lower()
        filtered = [
            q
            for q in filtered
            if sq in q.question.lower()
            or sq in q.gold_answer.lower()
            or (q.agent_answer and sq in q.agent_answer.lower())
        ]

    st.caption(f"Showing **{len(filtered)}** of **{len(all_questions)}** questions")

    if not filtered:
        st.info("No questions match the current filters.")
        return

    # 5. Question Selector Table / List
    def _format_q_option(q: BenchQuestionDetail) -> str:
        status_icon = (
            "🟢"
            if q.correctness == "correct"
            else ("🟡" if q.correctness == "partial" else ("🔴" if q.correctness == "incorrect" else "⚪"))
        )
        doc_short = Path(q.doc_name).stem if q.doc_name else "-"
        return f"{status_icon} [{q.id}] {doc_short}: {q.question[:75]}..."

    col_select, col_meta = st.columns([8, 2])
    with col_select:
        selected_idx = st.selectbox(
            "Select Question to Inspect",
            options=range(len(filtered)),
            format_func=lambda i: _format_q_option(filtered[i]),
            help="Choose a question to inspect its reference gold answer, agent output, judge evaluation, and execution trace",
        )
    with col_meta:
        st.metric("Filtered Set", len(filtered))

    selected_q = filtered[selected_idx]

    st.divider()

    # 6. Detailed Question Inspection View
    _render_question_detail(selected_q)


# ---------------------------------------------------------------------------
# Detail View Renderers
# ---------------------------------------------------------------------------


def _render_question_detail(q: BenchQuestionDetail) -> None:
    """Render the full comparison and trace for a single benchmark question."""
    # Status Header Banner
    status_color = (
        "#0f5132"
        if q.correctness == "correct"
        else ("#664d03" if q.correctness == "partial" else ("#842029" if q.correctness == "incorrect" else "#41464b"))
    )
    status_bg = (
        "#d1e7dd"
        if q.correctness == "correct"
        else ("#fff3cd" if q.correctness == "partial" else ("#f8d7da" if q.correctness == "incorrect" else "#e2e3e5"))
    )
    status_text = q.status_label

    st.markdown(
        f"""
        <div style="background-color: {status_bg}; border-left: 6px solid {status_color}; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px;">
            <div style="font-size: 1.2em; font-weight: bold; color: {status_color};">
                {status_text} · [{q.id}]
            </div>
            <div style="font-size: 0.9em; color: #333; margin-top: 4px;">
                📄 <strong>Manual:</strong> <code>{q.doc_name or "-"}</code> · 🛠️ <strong>Tool Calls:</strong> {q.n_tool_calls} · 🔤 <strong>Tokens:</strong> In: {q.input_tokens:,} / Out: {q.output_tokens:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Card 1: Question and Gold Reference Answer
    with st.container():
        st.markdown("#### ❓ Question")
        st.info(q.question)

        col_gold, col_agent = st.columns(2, gap="large")

        with col_gold:
            st.markdown("#### 🎯 Gold Reference Answer")
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 14px; min-height: 180px;">
                    {q.gold_answer}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if q.justification or q.evidence:
                with st.expander("Show Gold Evidence & Justification"):
                    if q.justification:
                        st.markdown(f"**Justification:** {q.justification}")
                    if q.evidence:
                        st.markdown(f"**Evidence:** `{q.evidence}`")

        with col_agent:
            st.markdown(f"#### 🤖 Agent Response ({q.run_llm or 'Model'})")
            if q.agent_answer:
                st.markdown(
                    f"""
                    <div style="background-color: #f0f7ff; border: 1px solid #b8daff; border-radius: 6px; padding: 14px; min-height: 180px;">
                        {q.agent_answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No agent answer recorded for this question.")

            if q.agent_thinking:
                with st.expander("💭 Agent Internal Reasoning / Thinking"):
                    st.markdown(q.agent_thinking)

    # Card 2: Judge Evaluation Breakdown
    st.divider()
    st.markdown("#### ⚖️ LLM Judge Evaluation")

    c_verdict, c_ground, c_num, c_cat = st.columns(4)
    c_verdict.metric("Correctness", (q.correctness or "N/A").upper())
    c_ground.metric("Groundedness", (q.groundedness or "N/A").title())
    c_num.metric("Numeric Match", "Yes" if q.numeric_match else ("No" if q.numeric_match is False else "N/A"))
    c_cat.metric("Error Category", q.error_category or "None")

    if q.rationale:
        st.markdown(
            f"""
            <div style="background-color: #fff9e6; border-left: 4px solid #ffc107; padding: 12px; border-radius: 4px; margin-top: 10px;">
                <strong>Judge Rationale ({q.judge_llm or "Judge"}):</strong><br/>
                {q.rationale}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 3: Execution Trace & Trajectory
    st.divider()
    st.markdown("#### 🔍 Execution Trace & Tool Trajectory")

    # Check for DecisionRAG navigation metadata
    nav_data = q.metadata.get("navigation") if isinstance(q.metadata, dict) else None
    if nav_data and isinstance(nav_data, dict):
        with st.expander("🧭 DecisionRAG Navigation Evidence Pack", expanded=True):
            col_n1, col_n2, col_n3, col_n4 = st.columns(4)
            col_n1.metric("Decision Model Calls", nav_data.get("dm_calls", 0))
            col_n2.metric("Inlined Sections", nav_data.get("pack_inlined", 0))
            col_n3.metric("Section Pointers", nav_data.get("pack_pointers", 0))
            col_n4.metric("Latency", f"{nav_data.get('elapsed_ms', 0) / 1000.0:.2f}s")

            levels = nav_data.get("levels") or []
            if levels:
                st.markdown("**Navigation Decision Levels & Scores:**")
                for lvl in levels:
                    st.write(
                        f"- **Level {lvl.get('level')}**: evaluated `{lvl.get('n_candidates')}` candidates. Selected scores: `{lvl.get('scores')}`"
                    )

    # Step-by-step tool calls
    if q.tool_calls:
        st.caption(f"Recorded **{len(q.tool_calls)}** tool calls during execution:")
        for idx, tc in enumerate(q.tool_calls):
            t_name = tc.get("name") or tc.get("tool") or f"Tool {idx + 1}"
            t_args = tc.get("args") or tc.get("arguments") or {}

            # Match tool result if available
            t_res = None
            if idx < len(q.tool_results):
                t_res = q.tool_results[idx]

            icon = "🛠️"
            if "toc" in t_name.lower():
                icon = "🗂️"
            elif "section" in t_name.lower():
                icon = "📑"
            elif "navigate" in t_name.lower():
                icon = "🧭"
            elif "image" in t_name.lower():
                icon = "🖼️"
            elif "read" in t_name.lower():
                icon = "📖"

            with st.expander(f"{icon} Step {idx + 1}: `{t_name}`", expanded=(idx < 2)):
                st.markdown("**Arguments:**")
                if isinstance(t_args, dict):
                    st.json(t_args)
                else:
                    st.code(str(t_args))

                if t_res is not None:
                    st.markdown("**Result Output:**")
                    res_str = (
                        t_res
                        if isinstance(t_res, str)
                        else json.dumps(t_res, indent=2)
                        if isinstance(t_res, dict)
                        else str(t_res)
                    )
                    if len(res_str) > 3000:
                        st.text_area(
                            f"Result (truncated, {len(res_str):,} chars)",
                            value=res_str[:3000] + "\n... [truncated]",
                            height=200,
                            key=f"res_{q.id}_{idx}",
                        )
                    else:
                        st.code(res_str)
    else:
        st.info("No intermediate tool calls recorded for this question.")
