"""Shared KG configuration selector widget for Streamlit pages.

Centralises the repeated ``get_available_kg_configs`` / selectbox /
``get_kg_manager.invalidate`` / ``st.rerun`` pattern found in every KG page.

Usage::

    from genai_graph.webapp.ui_components.kg_config_selector import (
        get_available_kg_configs,
        init_kg_config_session_state,
        render_kg_config_selector,
        render_schema_status,
    )

    # In your page's init block:
    init_kg_config_session_state()

    # In your sidebar rendering:
    render_kg_config_selector(help="Select a KG profile to inspect.")
    render_schema_status()  # optional — shows schema path + status

    # If you need page-specific reset logic on config change, pass a callback:
    render_kg_config_selector(on_change=lambda: setattr(sss, "my_cache", None))
"""

from collections.abc import Callable

import streamlit as st
from genai_tk.config_mgmt.config_mngr import global_config
from loguru import logger
from streamlit import session_state as sss

from genai_graph.kg.manager import get_kg_manager


def get_available_kg_configs() -> list[str]:
    """Return the sorted list of KG configuration names defined in ekg_workflows.yaml."""
    try:
        manager = get_kg_manager()
        return sorted(manager.ekg_config.kg_configs.keys())
    except Exception as exc:
        logger.warning("Could not load KG configurations: {}", exc)
        return ["default"]


def init_kg_config_session_state() -> None:
    """Seed ``sss.kg_config_selected`` from the available KG configurations.

    Picks the first available KG config so the page always starts with a
    valid profile — even when no ``kg_config`` is set in the global config.
    Safe to call multiple times — only initialises on first call.
    """
    if "kg_config_selected" not in sss or not sss.kg_config_selected:
        try:
            manager = get_kg_manager()
            available = sorted(manager.ekg_config.kg_configs.keys())
            profile = manager.profile
            sss.kg_config_selected = profile if profile in available else (available[0] if available else "default")
        except Exception:
            sss.kg_config_selected = "default"


def render_kg_config_selector(
    *,
    help: str = "Select the Knowledge Graph configuration to use.",
    on_change: Callable[[], None] | None = None,
) -> None:
    """Render the KG configuration selectbox and handle profile changes.

    Must be called inside a ``with st.sidebar:`` block (or at top level when
    already inside the sidebar context).  On selection change this function:

    1. Updates ``sss.kg_config_selected``
    2. Pushes the new value into ``global_config()`` (``kg_config`` key)
    3. Invalidates the ``get_kg_manager`` singleton
    4. Calls *on_change()* if provided (use for page-specific state resets)
    5. Calls ``st.rerun()``

    Args:
        help: Tooltip text for the selectbox.
        on_change: Optional callback for page-specific reset logic.
    """
    available_configs = get_available_kg_configs()
    current = sss.get("kg_config_selected", "default")
    current_index = available_configs.index(current) if current in available_configs else 0

    selected = st.selectbox(
        "⚙️ KG Configuration",
        options=available_configs,
        index=current_index,
        help=help,
        key="_kg_config_selector_widget",
    )

    if selected != current:
        sss.kg_config_selected = selected
        global_config().set("kg_config", selected)
        get_kg_manager.invalidate()  # pyright: ignore[reportFunctionMemberAccess]
        if on_change is not None:
            on_change()
        st.rerun()


def render_schema_status() -> None:
    """Display the schema file path and existence status for the selected KG config.

    Shows a success badge when the ``.txt`` schema file exists, or a warning
    with instructions to regenerate it.
    """
    try:
        manager = get_kg_manager()
        schema_path = manager.get_schema_path_for(sss.get("kg_config_selected", "default"))
        if schema_path.exists():
            st.success("✅ Schema loaded")
            st.caption(f"Path: {schema_path}")
        else:
            st.warning("⚠️ Schema not found. Run `cli kg schema --regen` first.")
    except Exception as exc:
        st.error(f"❌ Error loading KG schema: {exc}")
