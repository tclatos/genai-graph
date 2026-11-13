"""Streamlit page for interactive Benchmark Outcomes exploration.

Usage:
    cli bench web
    uv run streamlit run genai_graph/ui/streamlit/pages/demos/bench_browser.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from genai_graph.bench.config import load_bench_profile
from genai_graph.ui.streamlit.components.bench_browser import render_bench_browser


def main() -> None:
    """Main entry point for the standalone benchmark browser page."""
    st.set_page_config(
        page_title="Benchmark Outcomes Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    profile = os.getenv("BENCH_PROFILE")
    config_path = os.getenv("BENCH_CONFIG_PATH")
    cfg_p = Path(config_path) if config_path else None

    cfg = None
    try:
        cfg = load_bench_profile(profile_name=profile, config_path=cfg_p)
    except Exception as exc:
        st.sidebar.caption(f"Note: Profile '{profile}' fallback: {exc}")

    render_bench_browser(cfg=cfg, profile_name=profile, config_path=cfg_p)


if __name__ == "__main__":
    main()
